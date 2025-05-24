import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as T

from utils import build_graph, place_patches, normalize_positions, stitch_image, edge_strip, normalized_cross_correlation

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
resnet_model.eval()

# Transformation pipeline to prepare images for ResNet feature extraction
transform = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(imgs):
    # Convert input images to tensors, preprocess, and extract deep features using ResNet
    batch = torch.stack([transform(Image.fromarray(img)) for img in imgs]).to(device)
    with torch.no_grad():
        features = resnet_model(batch).squeeze(-1).squeeze(-1)
    return features.cpu().numpy() 

def compute_similarity(patches, thickness=10, weight_edge=0.6, weight_feat=0.4):
    N = len(patches)
    sides = ['right', 'left', 'bottom', 'top']
    sim = {s: np.full((N, N), -1.0) for s in sides}
    
    # Precompute all edge strips for efficiency
    edge_data = {
        s: [edge_strip(p, s, thickness) for p in patches]
        for s in sides
    }
    
    features = extract_features(patches)  # Extract deep features from all patches

    for i in tqdm(range(N), desc="Computing similarities"):
        for j in range(N):
            if i == j:
                continue
            
            # Compute all edge similarities at once
            edge_sim = {
                'right': normalized_cross_correlation(edge_data['right'][i], edge_data['left'][j]),
                'left': normalized_cross_correlation(edge_data['left'][i], edge_data['right'][j]),
                'bottom': normalized_cross_correlation(edge_data['bottom'][i], edge_data['top'][j]),
                'top': normalized_cross_correlation(edge_data['top'][i], edge_data['bottom'][j])
            }

            feat_sim = np.dot(features[i], features[j]) / (
                np.linalg.norm(features[i]) * np.linalg.norm(features[j]) + 1e-6)
            
            for s in sides:
                sim[s][i, j] = weight_edge * edge_sim[s] + weight_feat * feat_sim
    return sim

# Load the Images
def load_images(folder, resize_to=(224, 224)):
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))])
    if not files:
        raise ValueError(f"No image files found in folder '{folder}'.")
    images = []
    for f in files:
        img = Image.open(os.path.join(folder, f)).convert('RGB')
        img = img.resize(resize_to, Image.LANCZOS) 
        images.append(np.array(img)) 
    return images

# Main
def main():
    # Parse and process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='Path to folder of patches')
    parser.add_argument('-o', '--output', type=str, default='output.jpg', help='Output image file')
    args = parser.parse_args()

    if not os.path.exists(args.folder):
        print(f"Folder '{args.folder}' does not exist.")
        return
    
    print(f"Loading patches from {args.folder}...")
    try:
        patches = load_images(args.folder)
    except ValueError as ve:
        print(f"Error: {ve}")
        return

    print(f"Found {len(patches)} patches")

    sim = compute_similarity(patches)  
    G = build_graph(sim, top_k=2, threshold=0.7)  

    positions = place_patches(G, len(patches), sim) 
    normalized = normalize_positions(positions)  

    final_img = stitch_image(patches, normalized) 
    final_img.save(args.output)
    print(f"Stitched image saved to {args.output}")

if __name__ == '__main__':
    main()
