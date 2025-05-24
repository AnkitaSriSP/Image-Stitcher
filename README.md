# Image Stitcher

This project reconstructs an image from shuffled image patches using a hybrid approach, combining **deep feature similarity** (via ResNet18) and **edge strip matching** (using normalized cross-correlation), and then uses a graph-based method to determine patch layout.

---

## Quick Start (with Docker)

### 1. Build the Docker Image

```bash
docker build -t image-stitcher .
````

### 2. Run the Stitching Process

```bash
docker run --rm -v $(pwd):/data image-stitcher --folder <FOLDER_WITH_IMAGES> -o output.jpg
```

> Add `--gpus all` to the run command if using a GPU (NVIDIA CUDA supported).

---

## Command-Line Usage

```bash
python main.py --folder <FOLDER_WITH_IMAGES> -o <output_file>
```
---

## Sample Tests Provided

Two sample tests are included for evaluation:

* `image1.jpg`
  Patches in folder `test1` forming a 3x3 grid.

* `image2.jpg`
  Patches in folder `test2` forming a 1x5 grid.

You can test reconstruction using:

```bash
docker run --rm -v $(pwd):/data image-stitcher --folder test1 -o result1.jpg 
```
---

## Challenges Faced

While the system performs well in many situations, the following cases are more difficult:

* **Repetitive patterns**: Textures or tiles that repeat can confuse both deep features and edge matching.
* **Weak edges or noisy borders**: Inaccurate edge similarity scores can lead to misplacements.
* **Low contrast areas**: Deep features may become nearly identical, reducing discriminative power.

---

## Output

* The final reassembled image is saved to the file specified by `-o`.
