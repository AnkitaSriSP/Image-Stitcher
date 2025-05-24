import networkx as nx
from collections import deque
import numpy as np
from PIL import Image

DIRECTIONS = ['right', 'left', 'bottom', 'top']
OPPOSITE = {'right': 'left', 'left': 'right', 'top': 'bottom', 'bottom': 'top'}
DIR_OFFSET = {'right': (1, 0), 'left': (-1, 0), 'top': (0, -1), 'bottom': (0, 1)}

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

def build_graph(sim, top_k=1, threshold=0.7):
    N = sim['right'].shape[0]
    G = nx.DiGraph()

    for d in DIRECTIONS:
        for i in range(N):
            top_neighbors = np.argsort(sim[d][i])[::-1][:top_k]
            for j in top_neighbors:
                if i != j and sim[d][i, j] > threshold:
                    G.add_edge((i, d), j, weight=sim[d][i, j])
    return G

def get_best_seed(sim):
    N = sim['right'].shape[0]
    total_scores = np.zeros(N)
    for d in DIRECTIONS:
        total_scores += np.sum(np.maximum(sim[d], 0), axis=1)
    return int(np.argmax(total_scores))

def add_remaining_patches(placed, used, sim, N, threshold=0):
    position_map = {pos: idx for idx, pos in placed.items()}
    unplaced = set(range(N)) - used

    # Compute bounding box of placed patches
    xs = [pos[0] for pos in placed.values()]
    ys = [pos[1] for pos in placed.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    added_any = True
    while unplaced and added_any:
        added_any = False

        # Find all empty neighbor positions around placed patches
        empty_positions = set()
        for (x, y) in placed.values():
            for d in DIRECTIONS:
                dx, dy = DIR_OFFSET[d]
                new_pos = (x + dx, y + dy)
                if new_pos not in position_map:
                    empty_positions.add(new_pos)

        # If no empty spots, expand bounding box and add empty spots there
        if not empty_positions:
            min_x -= 1
            max_x += 1
            min_y -= 1
            max_y += 1
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    if (x, y) not in position_map:
                        empty_positions.add((x, y))

        placements = []  # (score, patch, pos)

        for patch in unplaced:
            best_score = -1
            best_pos = None

            for pos in empty_positions:
                # Check neighbors of pos to evaluate similarity
                neighbors = []
                for d in DIRECTIONS:
                    dx, dy = DIR_OFFSET[d]
                    neighbor_pos = (pos[0] - dx, pos[1] - dy)  # reverse offset to find neighbor
                    if neighbor_pos in position_map:
                        neighbors.append((neighbor_pos, d))
                # Get max similarity between patch and neighbors
                score = 0
                for neighbor_pos, d in neighbors:
                    neighbor_patch = position_map[neighbor_pos]
                    score = max(score, sim[d][neighbor_patch, patch])
                if score > best_score:
                    best_score = score
                    best_pos = pos

            if best_score >= threshold and best_pos is not None:
                placements.append((best_score, patch, best_pos))

        # Sort by descending score to place best matches first
        placements.sort(key=lambda x: x[0], reverse=True)

        # Place patches greedily
        for score, patch, pos in placements:
            if patch in unplaced and pos not in position_map:
                placed[patch] = pos
                used.add(patch)
                position_map[pos] = patch
                unplaced.remove(patch)
                empty_positions.remove(pos)
                added_any = True

        # If no patches placed, place any leftover patches arbitrarily
        if not added_any and unplaced and empty_positions:
            for patch, pos in zip(list(unplaced), list(empty_positions)):
                placed[patch] = pos
                used.add(patch)
                position_map[pos] = patch
                unplaced.remove(patch)
                empty_positions.remove(pos)
                added_any = True
                if not unplaced or not empty_positions:
                    break

def place_patches(G, N, sim):
    seed = get_best_seed(sim)
    placed = dict()
    used = set()
    queue = deque()

    placed[seed] = (0, 0)
    used.add(seed)
    queue.append(seed)

    position_map = dict()  # map position to (patch_idx, similarity_score)
    position_map[(0, 0)] = (seed, float('inf'))  # seed has infinite confidence

    while queue:
        current = queue.popleft()
        cur_pos = placed[current]

        for direction in DIRECTIONS:
            key = (current, direction)
            if key in G:
                for _, neighbor, data in G.edges(key, data=True):
                    if neighbor in used:
                        continue

                    dx, dy = DIR_OFFSET[direction]
                    new_pos = (cur_pos[0] + dx, cur_pos[1] + dy)

                    edge_sim_score = data['weight']

                    # Conflict resolution: replace if new similarity better
                    if new_pos in position_map:
                        _, existing_score = position_map[new_pos]
                        if edge_sim_score <= existing_score:
                            continue  # Existing patch placement is better, skip
                        else:
                            # Replace with better patch
                            replaced_patch = position_map[new_pos][0]
                            used.remove(replaced_patch)
                            placed.pop(replaced_patch)
                            position_map[new_pos] = (neighbor, edge_sim_score)
                            placed[neighbor] = new_pos
                            used.add(neighbor)
                            queue.append(neighbor)
                    else:
                        placed[neighbor] = new_pos
                        used.add(neighbor)
                        queue.append(neighbor)
                        position_map[new_pos] = (neighbor, edge_sim_score)

    # After BFS placement, add any remaining patches optimally
    add_remaining_patches(placed, used, sim, N, threshold=0)

    return placed

def normalize_positions(placed):
    xs = [x for x, y in placed.values()]
    ys = [y for x, y in placed.values()]
    min_x, min_y = min(xs), min(ys)
    return {k: (x - min_x, y - min_y) for k, (x, y) in placed.items()}

def stitch_image(patches, positions):
    patch_h, patch_w = patches[0].shape[:2]
    coords = list(positions.values())
    xs, ys = zip(*coords)
    min_x, min_y = min(xs), min(ys)
    max_x, max_y = max(xs), max(ys)

    grid_w = max_x - min_x + 1
    grid_h = max_y - min_y + 1

    stitched = Image.new('RGB', (grid_w * patch_w, grid_h * patch_h))

    for idx, (x, y) in positions.items():
        px, py = (x - min_x) * patch_w, (y - min_y) * patch_h
        patch_img = Image.fromarray(patches[idx])
        stitched.paste(patch_img, (px, py))

    return stitched
