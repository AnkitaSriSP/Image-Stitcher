# Image Stitcher

This project reconstructs an image from shuffled image patches using a hybrid approach, combining **deep feature similarity** (via ResNet18) and **edge strip matching** (using normalized cross-correlation), and then uses a graph-based method to determine patch layout.

It supports various layout styles and includes optional local optimization steps to improve reconstruction accuracy.

---

## Quick Start (with Docker)

### 1. Build the Docker Image

```bash
docker build -t image-stitcher .
````

### 2. Run the Stitching Process

```bash
docker run --rm -v $(pwd):/data image-stitcher patch1.jpg patch2.jpg patch3.jpg patch4.jpg \
--layout grid 2x2 -o output.jpg --patchsize 100x100 --edge_weight 0.6 --force 2
```

> Add `--gpus all` to the run command if using a GPU (NVIDIA CUDA supported).

---

## Command-Line Usage

```bash
python main.py <images>... --layout [horizontal | vertical | "grid RxC"] -o <output_file> [options]
```

### Arguments

| Argument        | Description                                                                           |
| --------------- | ------------------------------------------------------------------------------------- |
| `<images>`      | List of patch images (e.g., img1.jpg img2.jpg ...).                                   |
| `--layout`      | Layout type: `horizontal`, `vertical`, or `grid RxC` (e.g., `grid 3x3`).              |
| `-o, --output`  | Output filename (e.g., `stitched.jpg`).                                               |
| `--patchsize`   | Patch size, formatted as `WIDTHxHEIGHT` (e.g., `100x100`).                            |
| `--thickness`   | Edge strip thickness for comparison (default: `5`).                                   |
| `--edge_weight` | Weight for edge similarity (0.0–1.0). Feature similarity weight is `1 - edge_weight`. |
| `--force`       | Enable local optimization: `1` for one pass, `2` for two passes (recommended).        |

---

## Why Use `--force`?

The `--force` flag applies a **local brute-force refinement** to improve accuracy:

* `--force 1`: One pass of patch-swapping based on similarity with neighbors.
* `--force 2`: Two refinement passes (yields best results but time consuming).

> Improves layout quality in complex or ambiguous cases.

---

## Sample Tests Provided

Two sample test sets are included for evaluation:

* `image1.jpg`
  Patches: `patch1_1.jpg`, `patch1_2.jpg`, `patch1_3.jpg`, ..., forming a 3x3 grid.

* `image2.jpg`
  Patches: `patch2_1.jpg`, `patch2_2.jpg`, `patch2_3.jpg`, ..., forming a 4x4 grid.

You can test reconstruction using:

```bash
docker run --rm -v $(pwd):/data image-stitcher patch1_*.jpg --layout grid 2x2 -o result1.jpg --force 2
```

Adjust grid dimensions and patch size as per need.

---

## Challenges Faced

While the system performs well in many situations, the following cases are more difficult:

* **Repetitive patterns**: Textures or tiles that repeat can confuse both deep features and edge matching.
* **Weak edges or noisy borders**: Inaccurate edge similarity scores can lead to misplacements.
* **Low contrast areas**: Deep features may become nearly identical, reducing discriminative power.

Using `--force 2` significantly improves layout robustness in such situations.

---

## Output

* The final reassembled image is saved to the file specified by `-o`.
