# Image-Stitcher
This project reassembles/stitches scrambled image patches into a full image using a combination of deep features (from ResNet18) and edge similarity metrics. It's designed for puzzle-solving or image reconstruction tasks.
---

## Features

- Deep feature extraction using a pretrained **ResNet18**.
- Edge similarity using **Normalized Cross-Correlation**.
- Smart patch placement using similarity-based greedy matching.
- Supports layouts: `horizontal`, `vertical`, or custom `grid RxC` (e.g., `grid 3x4`).
- Docker-ready.

---

### Requirements

- Python 3.8+
- PyTorch
- torchvision
- PIL (Pillow)
- numpy
- tqdm

Install them manually or use the included Dockerfile (recommended).

---

## Docker Usage

### 1. Build the Docker image

```bash
docker build -t image-stitcher .
````

### 2. Run the stitcher

```bash
docker run --rm -v $(pwd):/data image-stitcher \
    img1.jpg img2.jpg img3.jpg img4.jpg \
    --layout "grid 2x2" \
    -o result.jpg
```
---

## Usage without Docker

```bash
python main.py image1.jpg image2.jpg ... imageN.jpg \
  --layout horizontal|vertical|grid RxC \
  -o output.jpg \
  [--threshold 0.5] \
  [--patchsize 100x100] \
  [--thickness 5] \
  [--edge_weight 0.5]
```

### Arguments

| Argument         | Description                                                                    |
| ---------------- | ------------------------------------------------------------------------------ |
| `images`         | List of input image patches.                                                   |
| `--layout`       | One of: `horizontal`, `vertical`, or `grid RxC` (e.g. `grid 2x3`). Required.   |
| `-o`, `--output` | Output path for the stitched image. Required.                                  |
| `--threshold`    | Similarity threshold to accept patch adjacency (default: 0.5).                 |
| `--patchsize`    | Resize all patches to `WIDTHxHEIGHT` before processing (default: 100x100).     |
| `--thickness`    | Thickness of the strip (in pixels) used for edge comparison (default: 5).      |
| `--edge_weight`  | Relative weight of edge similarity vs. deep feature similarity (default: 0.4). |

---

## How It Works

1. **Preprocessing**: Resizes all patches to a uniform size.
2. **Feature Extraction**: Deep features are extracted using a truncated ResNet18.
3. **Edge Similarity**: Calculates similarity between patch borders using normalized cross-correlation.
4. **Grid Assembly**: Starts from a corner patch and greedily assembles neighbors based on similarity scores.
5. **Stitching**: The grid is rendered into a single image.

---

## Challenges Faced

* Ensuring accurate reassembly across varying grid sizes — while a lower similarity threshold (0.3 used here) performs well for larger grids by allowing more flexibility, it becomes less effective for smaller layouts (like 2×2), where the greedy strategy struggles to find optimal matches, often reducing overall accuracy.
* **No Rotation or Flip Handling**: This version assumes all patches are upright. It cannot handle rotated or flipped patches.
* **Greedy Assembly**: The reassembly process uses a greedy strategy. While fast, it does not guarantee globally optimal results. Incorrect early choices may lead to suboptimal final arrangements.

---

## File Structure

```
main.py          # Main script
Dockerfile       # Docker support
README.md        # This file
```

---

## Output

The output is a stitched image saved to the path specified with `-o` or `--output`.

