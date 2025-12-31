
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from cleanlab.filter import find_label_issues
from cleanlab.outlier import OutOfDistribution
from torch.utils.data import DataLoader
import pandas as pd
import shutil

import pandas as pd
import shutil
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class CorrectorGUI:
    def __init__(self, dataset, issues, pred_probs, labels):
        self.dataset = dataset
        self.issues = issues
        self.pred_probs = pred_probs
        self.labels = labels
        self.current_idx = 0
        self.changes_count = 0
        
        # Filter issues to only high confidence ones
        self.filtered_issues = []
        for idx in self.issues:
            pred_label = np.argmax(self.pred_probs[idx])
            conf = self.pred_probs[idx][pred_label]
            if conf >= 0.90:
                current_label = self.dataset.samples[idx][1]
                if current_label != pred_label:
                    self.filtered_issues.append(idx)
                    
        if not self.filtered_issues:
            print("No high-confidence (>90%) issues found for correction.")
            return

        self.root = tk.Tk()
        self.root.title("Cleanlab Label Corrector")
        self.root.geometry("800x600")
        
        self.label_info = tk.Label(self.root, text="", font=("Arial", 14))
        self.label_info.pack(pady=10)
        
        self.canvas = tk.Label(self.root)
        self.canvas.pack(expand=True)
        
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20, fill='x')
        
        self.btn_accept = tk.Button(btn_frame, text="Accept Fix (Green)", bg="#90EE90", height=2, width=20, command=self.accept_fix)
        self.btn_accept.pack(side=tk.LEFT, padx=20, expand=True)
        
        self.btn_skip = tk.Button(btn_frame, text="Skip (Red)", bg="#FFB6C1", height=2, width=20, command=self.skip)
        self.btn_skip.pack(side=tk.LEFT, padx=20, expand=True)

        self.btn_auto = tk.Button(btn_frame, text="Auto Fix All (Blue)", bg="#ADD8E6", height=2, width=20, command=self.auto_fix_all)
        self.btn_auto.pack(side=tk.LEFT, padx=20, expand=True)
        
        self.show_current()
        self.root.mainloop()

    def show_current(self):
        if self.current_idx >= len(self.filtered_issues):
            self.root.destroy()
            return
            
        idx = self.filtered_issues[self.current_idx]
        path, current_label_idx = self.dataset.samples[idx]
        current_class = self.dataset.classes[current_label_idx]
        
        pred_label_idx = np.argmax(self.pred_probs[idx])
        pred_class = self.dataset.classes[pred_label_idx]
        conf = self.pred_probs[idx][pred_label_idx]
        
        self.label_info.config(
            text=f"File: {os.path.basename(path)}\nGiven: {current_class}  -->  Predicted: {pred_class}\nConfidence: {conf:.2%}"
        )
        
        # Load Image
        try:
            img = Image.open(path)
            # Resize to fit window 
            img.thumbnail((600, 400))
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.config(image=self.photo)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            self.skip()

    def accept_fix(self):
        idx = self.filtered_issues[self.current_idx]
        pred_label_idx = np.argmax(self.pred_probs[idx])
        pred_class = self.dataset.classes[pred_label_idx]
        path, _ = self.dataset.samples[idx]
        
        self._move_file(idx, path, pred_class, pred_label_idx)
        self.current_idx += 1
        self.show_current()

    def skip(self):
        self.current_idx += 1
        self.show_current()
        
    def auto_fix_all(self):
        print("\nAuto-fixing remaining issues...")
        while self.current_idx < len(self.filtered_issues):
            idx = self.filtered_issues[self.current_idx]
            pred_label_idx = np.argmax(self.pred_probs[idx])
            pred_class = self.dataset.classes[pred_label_idx]
            path, _ = self.dataset.samples[idx]
            
            self._move_file(idx, path, pred_class, pred_label_idx)
            self.current_idx += 1
            
        print("Auto-fix complete.")
        self.root.destroy()

    def _move_file(self, idx, path, pred_class, pred_label_idx):
        try:
            file_name = os.path.basename(path)
            new_dir = os.path.join(self.dataset.root, pred_class)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            new_path = os.path.join(new_dir, file_name)
            
            if os.path.exists(new_path):
                print(f"File {file_name} already exists in {pred_class}. Skipping move.")
            else:
                shutil.move(path, new_path)
                # Update memory
                self.labels[idx] = pred_label_idx
                self.dataset.samples[idx] = (new_path, pred_label_idx)
                self.changes_count += 1
                print(f"Moved {file_name} to {pred_class}")
        except Exception as e:
            print(f"Failed to move file: {e}")

def interactive_fix(dataset, issue_indices, pred_probs, labels):
    app = CorrectorGUI(dataset, issue_indices, pred_probs, labels)
    # The app modifies 'changes_count' in its instance. We need to access it after loop.
    # But mainloop blocks. So when __init__ returns (if we didn't call mainloop there), we'd have control.
    # But we called mainloop inside __init__, so it blocks until destroy() is called.
    return app.changes_count

# Configuration
SOURCE_DATA_DIR = "dataset"  # Original read-only source
WORK_DIR = "dataset_cleaned" # Working copy
BATCH_SIZE = 32

def get_feature_extractor():
    print("Loading DINOv2 model...")
    # Load DINOv2 ViT-S/14 from Torch Hub
    # Returns a VisionTransformer
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def extract_features(model, dataloader, device):
    features_list = []
    labels_list = []
    
    print("Extracting features from images using DINOv2...")
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            # DINOv2 forward pass returns the CLS token embedding (B, 384)
            out = model(inputs)
            # Make sure it's on CPU and numpy
            features_list.append(out.cpu().numpy())
            labels_list.append(targets.numpy())
            
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels

def visualize_features(features, labels, class_names, filename='tsne.png'):
    print("Running t-SNE to visualize features...")
    # Reduce to 2 dimensions
    # Perplexity must be less than n_samples
    perplex = min(30, len(features)-1)
    if perplex < 1: perplex = 1
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplex)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    
    # Plot each class
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = np.where(labels == label)
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=class_name, alpha=0.7)
        
    plt.legend()
    plt.title("t-SNE Visualization of DINOv2 Features")
    # Clean up axis logs
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Visualization saved to {filename}")

def generate_html_report(dataset, label_issues_indices, outlier_indices, ood_scores, pred_probs, filename="report.html"):
    print(f"\nGenerating HTML report: {filename}")
    
    html_content = """
    <html>
    <head>
        <title>Cleanlab Analysis Report</title>
        <style>
            body { font-family: sans-serif; margin: 20px; }
            .section { margin-bottom: 40px; }
            .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px; }
            .card { border: 1px solid #ccc; padding: 10px; border-radius: 5px; background: #f9f9f9; }
            img { max-width: 100%; height: auto; border-radius: 3px; }
            h2 { color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
            .highlight { color: #d9534f; font-weight: bold; }
            .plot-container { text-align: center; margin-bottom: 30px; }
            .plot-container img { max-width: 800px; border: 1px solid #eee; }
        </style>
    </head>
    <body>
        <h1>Cleanlab Analysis Report</h1>
        
        <div class="section">
            <h2>Feature Visualization (t-SNE)</h2>
            <p>2D projection of DINOv2 embeddings. Points are colored by their <strong>given label</strong>.</p>
            <div class="plot-container">
                <img src="tsne.png" alt="t-SNE Plot">
            </div>
        </div>
    """

    # Section 1: Top Label Issues
    html_content += """
    <div class="section">
        <h2>Top Potential Label Errors</h2>
        <p>Images where the given label likely contradicts the content.</p>
        <div class="grid">
    """
    
    # Show top 20 label issues
    top_issues = label_issues_indices[:20]
    for idx in top_issues:
        path, given_label_idx = dataset.samples[idx]
        class_name = dataset.classes[given_label_idx]
        # Get predicted class (max prob)
        pred_label_idx = np.argmax(pred_probs[idx])
        pred_class_name = dataset.classes[pred_label_idx]
        prob = pred_probs[idx][pred_label_idx]
        
        # Use relative path for HTML
        rel_path = os.path.relpath(path, start=os.path.dirname(os.path.abspath(filename)))

        html_content += f"""
        <div class="card">
            <img src="{rel_path}" alt="Image {idx}">
            <p><strong>File:</strong> {os.path.basename(path)}</p>
            <p><strong>Given:</strong> {class_name}</p>
            <p><strong>Predicted:</strong> {pred_class_name} ({prob:.2f})</p>
        </div>
        """
    html_content += "</div></div>"

    # Section 2: Top Outliers
    html_content += """
    <div class="section">
        <h2>Top Outliers (Out-of-Distribution)</h2>
        <p>Images with the lowest confidence scores (most anomalous).</p>
        <div class="grid">
    """
    
    # Sort outliers by score (ascending) if not already, but outlier_indices passed might differ
    # We want the indices with lowest scores. 
    # Let's map scores to indices
    sorted_indices = np.argsort(ood_scores)
    # Take bottom 20
    top_outliers = sorted_indices[:20]
    
    for idx in top_outliers:
        path, given_label_idx = dataset.samples[idx]
        class_name = dataset.classes[given_label_idx]
        score = ood_scores[idx]
        rel_path = os.path.relpath(path, start=os.path.dirname(os.path.abspath(filename)))

        html_content += f"""
        <div class="card">
            <img src="{rel_path}" alt="Image {idx}">
            <p><strong>File:</strong> {os.path.basename(path)}</p>
            <p><strong>Class:</strong> {class_name}</p>
            <p><strong>OOD Score:</strong> {score:.4f}</p>
        </div>
        """
        
    html_content += """
        </div>
    </div>
    </body>
    </html>
    """
    
    with open(filename, "w") as f:
        f.write(html_content)
    print("Report saved.")

def main():
    # 1. Setup Data Copy
    if not os.path.exists(SOURCE_DATA_DIR):
        print(f"Error: Source directory '{SOURCE_DATA_DIR}' not found.")
        return

    if not os.path.exists(WORK_DIR):
        print(f"Creating working copy of dataset at '{WORK_DIR}'...")
        shutil.copytree(SOURCE_DATA_DIR, WORK_DIR)
    else:
        print(f"Using existing working copy at '{WORK_DIR}'")
        
    DATA_DIR = WORK_DIR # Use the copy for everything

    print(f"Loading images from {DATA_DIR}...")
    # DINOv2 transforms: Resize to multiples of 14, Normalize with ImageNet mean/std
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), # 224 is divisible by 14 (14*16)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

    # 2. Extract Features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = get_feature_extractor()
    model.to(device)
    
    features, labels = extract_features(model, dataloader, device)

    print(f"Feature matrix shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    # Visualization
    visualize_features(features, labels, dataset.classes, filename='tsne.png')

    # 3. Train Classifier & Get Probabilities
    print("\nTraining classifier (Neural Network) on embeddings...")
    # MLPClassifier: Neural Network with one hidden layer of size 512
    clf = MLPClassifier(hidden_layer_sizes=(512,), max_iter=1000, random_state=42)
    pred_probs = cross_val_predict(clf, features, labels, cv=5, method="predict_proba")
    
    # 4. Find Label Issues
    print("\nDetecting label issues with Cleanlab...")
    issue_indices = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence"
    )
    print(f"Found {len(issue_indices)} potential label issues.")
    
    # --- Interactive Correction Start ---
    if len(issue_indices) > 0:
        changes = interactive_fix(dataset, issue_indices, pred_probs, labels)
        
        if changes > 0:
            print("\n" + "="*50)
            print(f"Changes verified ({changes} files moved). Re-running analysis to verify fixes...")
            print("="*50 + "\n")
            
            # Re-train and Re-predict (using updated labels)
            # Note: We are using the SAME features. The assumption is features didn't change, just labels.
            print("Re-training classifier with updated labels...")
            clf = MLPClassifier(hidden_layer_sizes=(512,), max_iter=1000, random_state=42)
            pred_probs = cross_val_predict(clf, features, labels, cv=5, method="predict_proba")
            
            # Re-find issues
            print("Detecting label issues (Pass 2)...")
            issue_indices = find_label_issues(
                labels=labels,
                pred_probs=pred_probs,
                return_indices_ranked_by="self_confidence"
            )
            print(f"Found {len(issue_indices)} potential label issues after correction.")
    # --- Interactive Correction End ---

    # 5. Outlier Detection
    print("\nDetecting Outliers (OOD)...")
    ood = OutOfDistribution()
    ood_scores = ood.fit_score(features=features, labels=labels)
    
    threshold = np.percentile(ood_scores, 5)
    outlier_indices = np.where(ood_scores < threshold)[0]
    print(f"Found {len(outlier_indices)} potential outliers (bottom 5%).")

    # 6. Generate Report
    generate_html_report(dataset, issue_indices, outlier_indices, ood_scores, pred_probs)

if __name__ == "__main__":
    main()
