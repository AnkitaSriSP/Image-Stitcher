import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import networkx as nx
from utils import load_and_resize, edge_strip, normalized_cross_correlation, build_grid_brute, refine_grid_local

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
resnet_model.eval()

transform = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(imgs):
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
    
    features = extract_features(patches)

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
            
            # Compute feature similarity
            feat_sim = np.dot(features[i], features[j]) / (
                np.linalg.norm(features[i]) * np.linalg.norm(features[j]) + 1e-6)
            
            # Combine similarities
            for s in sides:
                sim[s][i, j] = weight_edge * edge_sim[s] + weight_feat * feat_sim
    return sim

def select_initial_patch(sim):
    N = sim['right'].shape[0]
    # Calculate how "border-like" each patch is
    border_scores = np.zeros(N)
    for i in range(N):
        # Count how many directions have low similarity (likely borders)
        border_scores[i] = sum(
            np.max(sim[dir][i, :]) < 0.5 
            for dir in ['top', 'left', 'right', 'bottom']
        )
    
    # Among patches with highest border score, pick the one with most uniform neighbors
    max_score = np.max(border_scores)
    candidates = np.where(border_scores == max_score)[0]
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Select candidate with most uniform neighbor similarities
    uniformity_scores = []
    for c in candidates:
        all_sims = np.concatenate([sim[d][c, :] for d in ['top', 'left', 'right', 'bottom']])
        uniformity_scores.append(-np.std(all_sims))  # Lower std is better
        
    return candidates[np.argmax(uniformity_scores)]

def force_directed_dir(from_idx, to_idx, sim):
    similarities = {d: sim[d][from_idx][to_idx] for d in ['right', 'left', 'top', 'bottom']}
    best = max(similarities.items(), key=lambda x: x[1])[0]
    return {
        'right': (1, 0),
        'left': (-1, 0),
        'bottom': (0, 1),
        'top': (0, -1)
    }[best]


def build_grid(patches, sim, rows, cols):
    N = len(patches)
    positions = {}
    G = nx.Graph()

    # Build complete graph with directional similarities
    for direction, sim_matrix in sim.items():
        for i in range(N):
            for j in range(N):
                if i != j:
                    w = -sim_matrix[i][j]
                    if not G.has_edge(i, j) or w < G[i][j]['weight']:
                        G.add_edge(i, j, weight=w)

    mst = nx.minimum_spanning_tree(G, weight='weight')
    root = select_initial_patch(sim)

    # Modified force-directed placement with boundary checking
    def dfs(node, pos, visited=None):
        if visited is None:
            visited = set()
        if node in visited:
            return
        visited.add(node)
        positions[node] = pos
        
        for neighbor in mst.neighbors(node):
            if neighbor not in visited:
                offset = force_directed_dir(node, neighbor, sim)
                new_pos = (pos[0] + offset[0], pos[1] + offset[1])
                dfs(neighbor, new_pos, visited)

    dfs(root, (0, 0))

    # Normalize grid coordinates to start at (0,0)
    xs, ys = zip(*positions.values())
    min_x, min_y = min(xs), min(ys)
    normalized = {k: (x - min_x, y - min_y) for k, (x, y) in positions.items()}

    # Create grid and track used patches
    grid = np.full((rows, cols), -1, dtype=int)
    used_patches = set()
    available_positions = set((r, c) for r in range(rows) for c in range(cols))
    
    # First pass: place patches that fit perfectly in the grid
    for idx, (x, y) in normalized.items():
        if 0 <= x < cols and 0 <= y < rows and grid[y, x] == -1:
            grid[y, x] = idx
            used_patches.add(idx)
            available_positions.remove((y, x))

    # Second pass: fill remaining positions with best matches
    remaining_patches = set(range(N)) - used_patches
    
    while remaining_patches and available_positions:
        best_score = -1
        best_position = None
        best_patch = None
        
        # Find the best patch-position pair
        for (r, c) in available_positions:
            for patch_idx in remaining_patches:
                # Check all four neighbors
                total_score = 0
                neighbor_count = 0
                
                for dr, dc, side1, side2 in [(-1, 0, 'top', 'bottom'), (1, 0, 'bottom', 'top'),
                                           (0, -1, 'left', 'right'), (0, 1, 'right', 'left')]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                        neighbor_idx = grid[nr, nc]
                        total_score += sim[side1][neighbor_idx][patch_idx]
                        neighbor_count += 1
                
                if neighbor_count > 0:
                    avg_score = total_score / neighbor_count
                    if avg_score > best_score:
                        best_score = avg_score
                        best_position = (r, c)
                        best_patch = patch_idx
        
        if best_position is not None:
            grid[best_position[0], best_position[1]] = best_patch
            used_patches.add(best_patch)
            remaining_patches.remove(best_patch)
            available_positions.remove(best_position)
        else:
            # No good matches found - place random remaining patch
            r, c = available_positions.pop()
            grid[r, c] = remaining_patches.pop()

    return grid

def stitch_grid(patches, grid, patch_size):
    rows, cols = grid.shape
    out_img = Image.new('RGB', (cols * patch_size[0], rows * patch_size[1]))
    for r in range(rows):
        for c in range(cols):
            idx = grid[r, c]
            out_img.paste(Image.fromarray(patches[idx]), (c * patch_size[0], r * patch_size[1]))
    return out_img

def main():
    # Parse and process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('images', nargs='+')
    parser.add_argument('--layout', required=True, help="horizontal | vertical | grid RxC")
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--patchsize', default='100x100')
    parser.add_argument('--thickness', type=int, default=5)
    parser.add_argument('--edge_weight', type=float, default=0.4)
    parser.add_argument('-f', '--force', type=int, choices=[1, 2], help="Added local brute force, 1|2")
    args = parser.parse_args()

    layout = args.layout.lower().strip()
    if layout == 'horizontal':
        rows, cols = 1, len(args.images)
    elif layout == 'vertical':
        rows, cols = len(args.images), 1
    elif layout.startswith('grid'):
        try:
            _, rc = layout.split()
            rows, cols = map(int, rc.split('x'))
        except:
            print("Error: For grid layout, use format 'grid RxC'")
            return
    else:
        print("Error: --layout must be 'horizontal', 'vertical', or 'grid RxC'")
        return

    try:
        patch_w, patch_h = map(int, args.patchsize.split('x'))
    except:
        print("Error: --patchsize must be WIDTHxHEIGHT")
        return

    if rows * cols != len(args.images):
        print("Error: Layout does not match number of images")
        return

    try:
        patches = [load_and_resize(p, (patch_w, patch_h)) for p in args.images]
    except Exception as e:
        print(e)
        return

    patches = sorted(patches, key=lambda x: np.mean(x))
    print("Computing similarities...")
    sim = compute_similarity(patches, thickness=args.thickness,
                             weight_edge=args.edge_weight,
                             weight_feat=1 - args.edge_weight)

    print("Building grid layout...")
    if len(args.images)<=9:
        grid = build_grid_brute(patches, sim, rows, cols)
    else:
        grid = build_grid(patches, sim, rows, cols)
        # Additional local optimizations if any
        if args.force:
            print("First optimization pass...")
            grid = refine_grid_local(grid, sim)
            if args.force == 2:
                print("Second optimization pass...")
                grid = refine_grid_local(grid, sim)

    stitched = stitch_grid(patches, grid, (patch_w, patch_h))

    try:
        stitched.save(args.output)
        print(f"Saved stitched image to {args.output}")
    except Exception as e:
        print(f"Failed to save image: {e}")

if __name__ == "__main__":
    main()