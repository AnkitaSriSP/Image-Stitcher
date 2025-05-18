# Image-Stitcher
This project reassembles/stitches scrambled image patches into a full image using a combination of deep features (from ResNet18) and edge similarity metrics. It's designed for puzzle-solving or image reconstruction tasks.

---

## Features

- Deep feature extraction using a pretrained **ResNet18**.
- Edge similarity using **Normalized Cross-Correlation**.
- Smart patch placement using similarity-based greedy matching.
- Works on CPU and GPU.
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
    --grid 2x2 \
    -o result.jpg
```

Make sure the number of images matches `rows × cols` in the `--grid` argument.

---

## Manual Installation

```bash
pip install numpy==1.26.4 opencv-python==4.11.0.86 pillow==11.2.1 scipy==1.15.3 tqdm torchvision
```

---

## Usage

```bash
python main.py image1.jpg image2.jpg ... imageN.jpg \
    --grid ROWSxCOLS \
    -o output.jpg \
    [--threshold 0.6] \
    [--patchsize 100x100] \
    [--thickness 10] \
    [--edge_weight 0.4]
```

### Example

```bash
python main.py patch1.jpg patch2.jpg patch3.jpg patch4.jpg \
    --grid 2x2 \
    -o stitched.jpg
```

---

## How It Works

1. **Preprocessing**: Resizes all patches to a uniform size.
2. **Feature Extraction**: Deep features are extracted using a truncated ResNet18.
3. **Edge Similarity**: Calculates similarity between patch borders using normalized cross-correlation.
4. **Grid Assembly**: Starts from a corner patch and greedily assembles neighbors based on similarity scores.
5. **Stitching**: The grid is rendered into a single image.

---

## Challenges Faced

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



