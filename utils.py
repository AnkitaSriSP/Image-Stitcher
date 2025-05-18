import numpy as np
from PIL import Image
from tqdm import tqdm
import math
from itertools import permutations
from tqdm import tqdm

def load_and_resize(path, size=(100, 100)):
    try:
        img = Image.open(path).convert('RGB').resize(size)
        return np.array(img)
    except Exception as e:
        raise IOError(f"Error loading image {path}: {e}")

def edge_strip(img, side, thickness=10):
    return {
        'left': img[:, :thickness, :],
        'right': img[:, -thickness:, :],
        'top': img[:thickness, :, :],
        'bottom': img[-thickness:, :, :]
    }[side]

# Correlates edges of two patches
def normalized_cross_correlation(a, b):
    a_flat = a.reshape(-1).astype(np.float32)
    b_flat = b.reshape(-1).astype(np.float32)
    a_mean, b_mean = np.mean(a_flat), np.mean(b_flat)
    numerator = np.sum((a_flat - a_mean) * (b_flat - b_mean))
    denominator = np.sqrt(np.sum((a_flat - a_mean) ** 2) * np.sum((b_flat - b_mean) ** 2))
    return numerator / denominator if denominator != 0 else -1

# Compare all patch arrangements
def build_grid_brute(patches, sim, rows, cols):
    N = len(patches)
    if N != rows * cols:
        raise ValueError("Mismatch between patch count and grid dimensions.")

    best_score = float('inf')
    best_perm = None
    patch_indices = list(range(N))

    # Brute-force or greedy permutation scoring (warning: slow for N > 9)
    for perm in tqdm(permutations(patch_indices), total=math.factorial(N), desc="QAP permutations"):
        score = 0
        grid = np.array(perm).reshape((rows, cols))
        for r in range(rows):
            for c in range(cols):
                curr = grid[r, c]
                if c + 1 < cols:
                    right = grid[r, c + 1]
                    score -= sim['right'][curr][right]
                if r + 1 < rows:
                    bottom = grid[r + 1, c]
                    score -= sim['bottom'][curr][bottom]
        if score < best_score:
            best_score = score
            best_perm = perm

    return np.array(best_perm).reshape((rows, cols))

# Local optimization with 3x3 sliding window
def refine_grid_local(grid, sim):
    rows, cols = grid.shape
    new_grid = grid.copy()

    for r in range(rows - 2):
        for c in range(cols - 2):
            subgrid = new_grid[r:r+3, c:c+3]
            patch_indices = subgrid.flatten()
            best_score = float('inf')
            best_perm = None

            for perm in permutations(patch_indices):
                score = 0
                local_grid = np.array(perm).reshape((3, 3))
                for i in range(3):
                    for j in range(3):
                        curr = local_grid[i, j]
                        if j + 1 < 3:
                            right = local_grid[i, j + 1]
                            score -= sim['right'][curr][right]
                        if i + 1 < 3:
                            bottom = local_grid[i + 1, j]
                            score -= sim['bottom'][curr][bottom]
                if score < best_score:
                    best_score = score
                    best_perm = perm

            # Replace 3x3 region with best configuration
            new_grid[r:r+3, c:c+3] = np.array(best_perm).reshape((3, 3))

    return new_grid


