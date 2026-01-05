
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from cleanlab.filter import find_label_issues
from cleanlab.outlier import OutOfDistribution
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import time
import pandas as pd
import shutil

import pandas as pd
import shutil
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class CorrectorPanel(ttk.Frame):
    def __init__(self, parent, dataset, issues, pred_probs, labels):
        super().__init__(parent)
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
                    
        self._setup_ui()
        if not self.filtered_issues:
            self.lbl_info.config(text="No high-confidence issues found.")
            self._disable_controls()
        else:
            self.show_current()

    def _setup_ui(self):
        self.lbl_info = tk.Label(self, text="", font=("Arial", 14))
        self.lbl_info.pack(pady=10)
        
        self.canvas = tk.Label(self)
        self.canvas.pack(expand=True)
        
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=20, fill='x')
        
        self.btn_accept = tk.Button(btn_frame, text="Accept Fix (Green)", bg="#90EE90", height=2, width=20, command=self.accept_fix)
        self.btn_accept.pack(side=tk.LEFT, padx=20, expand=True)
        
        self.btn_skip = tk.Button(btn_frame, text="Skip (Red)", bg="#FFB6C1", height=2, width=20, command=self.skip)
        self.btn_skip.pack(side=tk.LEFT, padx=20, expand=True)

        self.btn_auto = tk.Button(btn_frame, text="Auto Fix All (Blue)", bg="#ADD8E6", height=2, width=20, command=self.auto_fix_all)
        self.btn_auto.pack(side=tk.LEFT, padx=20, expand=True)

    def _disable_controls(self):
        self.btn_accept.config(state="disabled")
        self.btn_skip.config(state="disabled")
        self.btn_auto.config(state="disabled")

    def show_current(self):
        if self.current_idx >= len(self.filtered_issues):
            self.lbl_info.config(text="All issues processed.")
            self.canvas.config(image='')
            return
            
        idx = self.filtered_issues[self.current_idx]
        path, current_label_idx = self.dataset.samples[idx]
        current_class = self.dataset.classes[current_label_idx]
        
        pred_label_idx = np.argmax(self.pred_probs[idx])
        pred_class = self.dataset.classes[pred_label_idx]
        conf = self.pred_probs[idx][pred_label_idx]
        
        self.lbl_info.config(
            text=f"File: {os.path.basename(path)}\nGiven: {current_class}  -->  Predicted: {pred_class}\nConfidence: {conf:.2%}"
        )
        
        try:
            img = Image.open(path)
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
        self.show_current()

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
                self.labels[idx] = pred_label_idx
                self.dataset.samples[idx] = (new_path, pred_label_idx)
                self.changes_count += 1
                print(f"Moved {file_name} to {pred_class}")
        except Exception as e:
            print(f"Failed to move file: {e}")

class ExplorerPanel(ttk.Frame):
    def __init__(self, parent, features, labels, dataset, class_names):
        super().__init__(parent)
        self.features = features
        self.labels = labels
        self.dataset = dataset
        self.class_names = class_names
        
        self.paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)
        
        self.left_frame = ttk.Frame(self.paned, width=800)
        self.paned.add(self.left_frame, weight=3)
        
        self.right_frame = ttk.Frame(self.paned, width=400)
        self.paned.add(self.right_frame, weight=1)
        
        self._setup_preview_panel()
        
    def render_plot(self):
        # We call this explicitly so it doesn't freeze startup if slow
        print("Rendering t-SNE plot...")
        # Reduce if needed, but we assume features passed are ready
        # Calculate t-SNE (cached ideally, but we re-run for simplicity or use existing logic)
        tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
        features_2d = tsne.fit_transform(self.features)
        
        self.fig = plt.Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.scatter = self.ax.scatter(
            features_2d[:, 0], features_2d[:, 1], 
            c=self.labels, cmap='tab10', alpha=0.7, picker=5
        )
        
        unique_labels = np.unique(self.labels)
        handles, _ = self.scatter.legend_elements()
        self.ax.legend(handles, [self.class_names[i] for i in unique_labels], title="Classes")
        self.ax.set_title("Click points to view details")
        self.ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

    def _setup_preview_panel(self):
        self.lbl_title = tk.Label(self.right_frame, text="Select a point...", font=("Arial", 16, "bold"))
        self.lbl_title.pack(pady=20)
        self.lbl_img = tk.Label(self.right_frame)
        self.lbl_img.pack(fill=tk.BOTH, expand=True, padx=10)
        self.lbl_details = tk.Label(self.right_frame, text="", font=("Arial", 10), justify=tk.LEFT)
        self.lbl_details.pack(pady=20, padx=10, fill=tk.X)

    def on_pick(self, event):
        ind = event.ind[0]
        path, label_idx = self.dataset.samples[ind]
        class_name = self.class_names[label_idx]
        file_name = os.path.basename(path)
        
        self.lbl_title.config(text=class_name)
        self.lbl_details.config(text=f"File: {file_name}\nIndex: {ind}")
        
        try:
            img = Image.open(path)
            display_width = self.right_frame.winfo_width() - 20
            if display_width < 100: display_width = 300
            aspect = img.height / img.width
            display_height = int(display_width * aspect)
            img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(img)
            self.lbl_img.config(image=self.photo)
        except Exception as e:
            print(f"Error loading image: {e}")

class CleanlabDashboard(tk.Tk):
    def __init__(self, dataset, issues, pred_probs, labels, features):
        super().__init__()
        self.title("Cleanlab Dashboard")
        self.geometry("1400x900")
        
        self.dataset = dataset
        self.issues = issues
        self.pred_probs = pred_probs
        self.labels = labels
        self.features = features
        
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Corrector
        self.corrector_tab = CorrectorPanel(self.notebook, dataset, issues, pred_probs, labels)
        self.notebook.add(self.corrector_tab, text="Fix Issues")
        
        # Tab 2: Explorer
        self.explorer_tab = ExplorerPanel(self.notebook, features, labels, dataset, dataset.classes)
        self.notebook.add(self.explorer_tab, text="Explore Data")
        
        # Bind tab change to render plot only when needed (or just render immediately)
        # Rendering immediately is simpler for now
        self.explorer_tab.render_plot()
        
    def get_changes_count(self):
        return self.corrector_tab.changes_count

# Configuration
SOURCE_DATA_DIR = "../dataset"  # Original read-only source
WORK_DIR = "dataset_cleaned" # Working copy
BATCH_SIZE = 32
CUSTOM_WEIGHTS_PATH = "custom_weights.pth" # Path to your pretrained .pth file

def get_feature_extractor():
    print("Loading DINOv2 model (ViT-L/14)...")
    # Load DINOv2 ViT-L/14 from Torch Hub
    # Returns a VisionTransformer
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
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

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def predict_cnn(model, dataloader, device):
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            
    return np.concatenate(all_probs, axis=0)

def get_cnn_model(num_classes, device):
    """
    Creates the ResNet50 model and loads custom weights if available.
    """
    # 1. Initialize Standard ResNet50
    # Use 'DEFAULT' (ImageNet) weights if no custom path, otherwise start raw (or ImageNet then overwrite)
    # We will start with ImageNet as a base regardless, then overwrite if custom exists
    model = models.resnet50(weights='DEFAULT')
    
    # 2. Modify specific layer (FC) to match number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # 3. Load Custom Weights if provided
    if os.path.exists(CUSTOM_WEIGHTS_PATH):
        print(f"Loading custom weights from {CUSTOM_WEIGHTS_PATH}...")
        try:
            state_dict = torch.load(CUSTOM_WEIGHTS_PATH, map_location=device)
            # Handle potential DataParallel wrapping ('module.' prefix)
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            
            # Load with strict=False to allow for head mismatch (transfer learning)
            # This allows loading a model trained on 1000 classes into our N-class model
            # keeping the backbone weights but ignoring the final FC layer if shapes differ.
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Missing keys (likely head): {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
                
        except Exception as e:
            print(f"Error loading custom weights: {e}")
            print("Falling back to ImageNet weights.")
    else:
        if CUSTOM_WEIGHTS_PATH and CUSTOM_WEIGHTS_PATH != "custom_weights.pth":
             print(f"Warning: Custom weights file '{CUSTOM_WEIGHTS_PATH}' not found. Using ImageNet defaults.")
    
    model = model.to(device)
    return model

def get_cnn_probs(dataset, num_classes, batch_size=32, num_epochs=10, k=5):
    print(f"\nStarting {k}-Fold Cross-Validation with ResNet50...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Placeholder for all probabilities. 
    # We need to map them back to original indices.
    n_samples = len(dataset)
    all_pred_probs = np.zeros((n_samples, num_classes))
    
    # We also need a way to track which indices were predicted (sanity check)
    predicted_indices = np.zeros(n_samples, dtype=bool)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(n_samples))):
        print(f"\nFold {fold+1}/{k}")
        
        # Create Subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # Initialize Fresh Model with Logic
        model = get_cnn_model(num_classes, device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        # Train Loop
        for epoch in range(num_epochs):
            start_time = time.time()
            loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            # Optional: Validate on val set for monitoring (skipping for speed, but good for debugging)
            print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f} - Acc: {acc:.4f} - Time: {time.time()-start_time:.1f}s")
            
        # Predict on Validation Fold
        print(f"  Predicting on validation set ({len(val_idx)} images)...")
        probs = predict_cnn(model, val_loader, device)
        
        # Store probabilities
        all_pred_probs[val_idx] = probs
        predicted_indices[val_idx] = True
        
    return all_pred_probs

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
    print("\nTraining classifier (ResNet50) on images...")
    # Using ResNet50 with Cross Validation to get clean probabilities
    pred_probs = get_cnn_probs(dataset, num_classes=len(dataset.classes), batch_size=BATCH_SIZE, num_epochs=5, k=5)
    
    # 4. Find Label Issues
    print("\nDetecting label issues with Cleanlab...")
    issue_indices = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence"
    )
    print(f"Found {len(issue_indices)} potential label issues.")
    
    # --- Interactive Dashboard Start ---
    print("\nLaunching Cleanlab Dashboard...")
    app = CleanlabDashboard(dataset, issue_indices, pred_probs, labels, features)
    app.mainloop() # Blocks until window closed
    
    changes = app.get_changes_count()
    if changes > 0:
        print("\n" + "="*50)
        print(f"Changes found ({changes} files moved). Re-running analysis...")
        # ... logic to re-run ...
        # [Existing logic follows]
         
        # Re-train and Re-predict (using updated labels)
        print("Re-training classifier with updated labels...")
        pred_probs = get_cnn_probs(dataset, num_classes=len(dataset.classes), batch_size=BATCH_SIZE, num_epochs=5, k=5)
        
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
