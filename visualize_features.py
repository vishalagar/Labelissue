
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
DATA_DIR = "dataset_cleaned"
BATCH_SIZE = 32
OUTPUT_FILE = "tsne.png"

def get_feature_extractor():
    print("Loading DINOv2 model for visualization...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
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

def visualize_tsne(features, labels, class_names):
    print("Running t-SNE dimensionality reduction...")
    # Reduce to 50 dim with PCA first if features are huge, but 384 is fine for direct t-SNE usually.
    # However, PCA initialization is good practice.
    
    tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
    features_2d = tsne.fit_transform(features)
    
    print(f"Plotting {features.shape[0]} points...")
    plt.figure(figsize=(12, 10))
    
    # Scatter plot
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    
    # Legend
    # Create a legend with class names
    handles = scatter.legend_elements()[0]
    # Ensure we don't exceed the number of handles if some classes are missing
    unique_labels = np.unique(labels)
    plt.legend(handles, [class_names[i] for i in unique_labels], title="Classes")
    
    plt.title("DINOv2 Feature Visualization (t-SNE)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {OUTPUT_FILE}")
    plt.show() # Attempt to show if supported, otherwise just saves

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
    
    visualize_tsne(features, labels, dataset.classes)

if __name__ == "__main__":
    main()
