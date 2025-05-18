import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

# Global setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
resnet_model.eval()

transform = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and process patches
def load_and_resize(path, size=(100, 100)):
    try:
        img = Image.open(path).convert('RGB').resize(size)
        return np.array(img)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {path}")
    except OSError as e:
        raise OSError(f"Error opening image {path}: {e}")

# Edge similarity via normalized cross-correlation
def edge_strip(img, side, thickness=10):
    if side == 'left':
        return img[:, :thickness, :]
    elif side == 'right':
        return img[:, -thickness:, :]
    elif side == 'top':
        return img[:thickness, :, :]
    elif side == 'bottom':
        return img[-thickness:, :, :]

def normalized_cross_correlation(a, b):
    a_flat = a.reshape(-1).astype(np.float32)
    b_flat = b.reshape(-1).astype(np.float32)
    a_mean = np.mean(a_flat)
    b_mean = np.mean(b_flat)
    numerator = np.sum((a_flat - a_mean) * (b_flat - b_mean))
    denominator = np.sqrt(np.sum((a_flat - a_mean)**2) * np.sum((b_flat - b_mean)**2))
    return numerator / denominator if denominator != 0 else -1

# Feature extraction via Resnet18
def extract_features(imgs):
    batch = torch.stack([transform(Image.fromarray(img)) for img in imgs]).to(device)
    with torch.no_grad():
        features = resnet_model(batch).squeeze(-1).squeeze(-1)
    return features.cpu().numpy()

def compute_edge_sim(p1, p2, thickness, side1, side2):
    return normalized_cross_correlation(edge_strip(p1, side1, thickness), edge_strip(p2, side2, thickness))

# Compute weighted similarity using feature similarity and edge similarity
def compute_similarity(patches, thickness=10, weight_edge=0.6, weight_feat=0.4):
    N = len(patches)
    sides = ['right', 'left', 'bottom', 'top']
    sim = {s: np.full((N, N), -1.0) for s in sides}

    features = extract_features(patches)

    # Compute edge similarity for each pair of images and orientation and corresponding weighted similarity
    for i in tqdm(range(N), desc="Computing similarities"):
        for j in range(N):
            if i == j:
                continue
            edge_sim = {
                'right': compute_edge_sim(patches[i], patches[j], thickness, 'right', 'left'),
                'left': compute_edge_sim(patches[i], patches[j], thickness, 'left', 'right'),
                'bottom': compute_edge_sim(patches[i], patches[j], thickness, 'bottom', 'top'),
                'top': compute_edge_sim(patches[i], patches[j], thickness, 'top', 'bottom')
            }
            feat_sim = np.dot(features[i], features[j]) / (np.linalg.norm(features[i]) * np.linalg.norm(features[j]) + 1e-6)
            for s in sides:
                sim[s][i, j] = weight_edge * edge_sim[s] + weight_feat * feat_sim

    return sim

def find_corner_patch(sim, N, threshold=0.6):
    scores = []
    for i in range(N):
        score = sum(np.max(sim[side][i, :]) < threshold for side in ['top', 'left', 'right', 'bottom'])
        scores.append(score)
    return np.argmax(scores)

# Build grid via greedy approach and computed similarities
def build_grid(patches, sim, rows, cols, edge_threshold=0.6):
    N = len(patches)
    grid = np.full((rows, cols), -1, dtype=int)
    used = set()

    # Find a corner patch (likely has low similarity on multiple sides)
    corner = find_corner_patch(sim, N, threshold=edge_threshold)
    grid[0, 0] = corner
    used.add(corner)

    # Fill the first row from left to right using right-edge similarity
    for c in range(1, cols):
        left = grid[0, c - 1]
        candidates = [(j, sim['right'][left, j]) for j in range(N) if j not in used and sim['right'][left, j] >= edge_threshold]
        if not candidates:
            print(f"No candidate found for position (0, {c})")
            continue
        best = max(candidates, key=lambda x: x[1])
        grid[0, c] = best[0]
        used.add(best[0])

    # Fill remaining rows top-to-bottom using bottom-edge similarity
    for r in range(1, rows):
        for c in range(cols):
            top = grid[r - 1, c]
            candidates = [(j, sim['bottom'][top, j]) for j in range(N) if j not in used and sim['bottom'][top, j] >= edge_threshold]
            if not candidates:
                print(f"No candidate found for position ({r}, {c})")
                continue
            best = max(candidates, key=lambda x: x[1])
            grid[r, c] = best[0]
            used.add(best[0])

    # Fill any remaining unassigned cells with leftover patches (fallback)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == -1:
                remaining = list(set(range(N)) - used)
                if remaining:
                    chosen = remaining.pop()
                    grid[r, c] = chosen
                    used.add(chosen)

    return grid

def stitch_grid(patches, grid, patch_size):
    rows, cols = grid.shape
    out_img = Image.new('RGB', (cols * patch_size[0], rows * patch_size[1]))
    for r in range(rows):
        for c in range(cols):
            idx = grid[r, c]
            patch_img = Image.fromarray(patches[idx])
            out_img.paste(patch_img, (c * patch_size[0], r * patch_size[1]))
    return out_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images', nargs='+')
    parser.add_argument('--layout', required=True, help="Layout type: 'horizontal', 'vertical', or 'grid RxC' (e.g. 'grid 3x4')")
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--threshold', type=float, default=0.3, help='Minimum edge similarity to accept patch adjacency')
    parser.add_argument('--patchsize', default='100x100', help='Patch size as WIDTHxHEIGHT (default: 100x100)')
    parser.add_argument('--thickness', type=int, default=5, help='Edge thickness for similarity comparison')
    parser.add_argument('--edge_weight', type=float, default=0.6, help='Weight for edge similarity (vs deep feature)')
    args = parser.parse_args()

    layout_arg = args.layout.lower().strip()
    num_images = len(args.images)

    if layout_arg == 'horizontal':
        rows, cols = 1, num_images
    elif layout_arg == 'vertical':
        rows, cols = num_images, 1
    elif layout_arg.startswith('grid'):
        try:
            _, grid_spec = layout_arg.split()
            rows, cols = map(int, grid_spec.split('x'))
        except Exception:
            print("Error: For grid layout, use format 'grid RxC', e.g. 'grid 3x4'")
            return
    else:
        print("Error: --layout must be 'horizontal', 'vertical', or 'grid RxC'")
        return

    try:
        patch_w, patch_h = map(int, args.patchsize.lower().split('x'))
    except ValueError:
        print("Error: --patchsize must be in the format WIDTHxHEIGHT, e.g. 100x100")
        return

    patch_size = (patch_w, patch_h)
    expected_count = rows * cols
    if expected_count != num_images:
        print(f"Error: Layout size ({rows}x{cols}={expected_count}) does not match number of images ({num_images})")
        return

    try:
        patches = [load_and_resize(p, patch_size) for p in args.images]
    except Exception as e:
        print(f"Image loading error: {e}")
        return

    print("All images loaded successfully. Computing similarities...")
    sim = compute_similarity(patches, thickness=args.thickness,
                             weight_edge=args.edge_weight,
                             weight_feat=1 - args.edge_weight)

    grid = build_grid(patches, sim, rows, cols, edge_threshold=args.threshold)
    result = stitch_grid(patches, grid, patch_size)

    try:
        result.save(args.output)
        print(f"Saved stitched image as {args.output}")
    except Exception as e:
        print(f"Error saving output image: {e}")

if __name__ == "__main__":
    main()
