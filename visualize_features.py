
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Configuration
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configuration
DATA_DIR = "dataset_cleaned"
BATCH_SIZE = 32
OUTPUT_FILE = "tsne.png"

class FeatureExplorerApp:
    def __init__(self, features, labels, dataset, class_names):
        self.features = features
        self.labels = labels
        self.dataset = dataset
        self.class_names = class_names
        
        self.root = tk.Tk()
        self.root.title("DINOv2 Feature Explorer")
        self.root.geometry("1400x900")
        
        # Split Pane
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Left Panel: Plot
        self.left_frame = ttk.Frame(self.paned_window, width=900)
        self.paned_window.add(self.left_frame, weight=3)
        
        # Right Panel: Image Preview
        self.right_frame = ttk.Frame(self.paned_window, width=400)
        self.paned_window.add(self.right_frame, weight=1)
        
        self._setup_plot()
        self._setup_preview_panel()
        
        self.root.mainloop()
        
    def _setup_plot(self):
        print("Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
        features_2d = tsne.fit_transform(self.features)
        
        self.fig = plt.Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Scatter with picker=True (5 points tolerance)
        self.scatter = self.ax.scatter(
            features_2d[:, 0], features_2d[:, 1], 
            c=self.labels, cmap='tab10', alpha=0.7, picker=5
        )
        
        # Legend
        handles, _ = self.scatter.legend_elements()
        unique_labels = np.unique(self.labels)
        self.ax.legend(handles, [self.class_names[i] for i in unique_labels], title="Classes")
        
        self.ax.set_title("Click points to view images")
        self.ax.grid(True, alpha=0.3)
        
        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Connect event
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        
        print("Plot ready.")

    def _setup_preview_panel(self):
        self.lbl_title = tk.Label(self.right_frame, text="Select a point...", font=("Arial", 16, "bold"))
        self.lbl_title.pack(pady=20)
        
        self.lbl_img = tk.Label(self.right_frame)
        self.lbl_img.pack(fill=tk.BOTH, expand=True, padx=10)
        
        self.lbl_details = tk.Label(self.right_frame, text="", font=("Arial", 12), justify=tk.LEFT)
        self.lbl_details.pack(pady=20, padx=10, fill=tk.X)

    def on_pick(self, event):
        ind = event.ind[0] # Get index of the first point clicked
        
        # Update details
        path, label_idx = self.dataset.samples[ind]
        class_name = self.class_names[label_idx]
        file_name = os.path.basename(path)
        
        self.lbl_title.config(text=class_name)
        self.lbl_details.config(text=f"File: {file_name}\nIndex: {ind}")
        
        # Load and show image
        try:
            img = Image.open(path)
            # Resize
            display_width = self.right_frame.winfo_width() - 20
            if display_width < 100: display_width = 300
            
            aspect = img.height / img.width
            display_height = int(display_width * aspect)
            
            img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(img)
            self.lbl_img.config(image=self.photo)
        except Exception as e:
            print(f"Error loading image: {e}")

def get_feature_extractor():
    print("Loading DINOv2 model for visualization (ViT-L/14)...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def extract_features(model, dataloader, device):
    features_list = []
    labels_list = []
    
    print("Extracting features...")
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            out = model(inputs)
            features_list.append(out.cpu().numpy())
            labels_list.append(targets.numpy())
            
            
            
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory '{DATA_DIR}' not found.")
        return

    # Transforms (same as main_images.py)
    transform = transforms.Compose([
    
    
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = get_feature_extractor()
    model.to(device)
    
    features, labels = extract_features(model, dataloader, device)
    
    # Launch App
    app = FeatureExplorerApp(features, labels, dataset, dataset.classes)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
