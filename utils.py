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

# Local optimization with sliding window 
def refine_grid_local(grid, sim):
    rows, cols = grid.shape
    new_grid = grid.copy()

    # Case 1: Standard 3x3
    if rows >= 3 and cols >= 3:
        a=3
        b=3
    # Case 2: Less than 3 rows, at least 3 columns (use 1x3 or 2x3)
    elif rows < 3 and cols >= 3:
        a=rows
        b=3
        rows=3
    # Case 3: Less than 3 columns, at least 3 rows (use 3x1 or 3x2)
    elif cols < 3 and rows >= 3:
        a=3
        b=cols
        cols=3

    for r in range(rows - 2):
        for c in range(cols - 2):
            subgrid = new_grid[r:r+a, c:c+b]
            patch_indices = subgrid.flatten()
            best_score = float('inf')
            best_perm = None
            shape=(a,b)

            for perm in permutations(patch_indices):
                score = 0
                local_grid = np.array(perm).reshape(shape)
                for i in range(a):
                    for j in range(b):
                        curr = local_grid[i, j]
                        if j + 1 < b:
                            right = local_grid[i, j + 1]
                            score -= sim['right'][curr][right]
                        if i + 1 < a:
                            bottom = local_grid[i + 1, j]
                            score -= sim['bottom'][curr][bottom]
                if score < best_score:
                    best_score = score
                    best_perm = perm

            new_grid[r:r+a, c:c+b] = np.array(best_perm).reshape((a, b))
    return new_grid


