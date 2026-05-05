# Arm Workspace

ROS2 workspace for the robot arm controller.

## Pretrained Model

The trained model can be downloaded here:
[Download model (Google Drive)](https://drive.google.com/file/d/1zWj5caLaHoNQTU14fD86SZdeKuiThD4Z/view?usp=drive_link)

Place it in `/segmentation_checkpoints/` before running.

# Water Detection

**Author:** Iyad Laphir

Real-time water segmentation and level detection using a custom neural network built on a ResNet18 encoder + U-Net decoder architecture.

## What it does

- **Segments** the water region in an image at the pixel level (blue overlay)
- **Classifies** the water fill level on a 0–4 scale (empty → full)
- **Runs live** from a webcam with press-to-analyse controls

## Files

| File | Description |
|---|---|
| `model.py` | `WaterSegNet` architecture — ResNet18 encoder + U-Net decoder + level classifier |
| `water_detection_2.py` | Training script — trains on COCO-annotated water data |
| `predict.py` | Inference on a single image or webcam frame |
| `webcam_detect.py` | Live webcam detection loop |

## How it works (beginner-friendly)

### CNNs — the foundation
A **Convolutional Neural Network (CNN)** is a type of neural network designed for images. Instead of looking at every pixel individually, it slides small filters across the image to detect patterns — edges, textures, shapes — building up a richer understanding with each layer.

### ResNet18 — the encoder
This project uses **ResNet18**, a well-known CNN architecture trained on millions of images (ImageNet). Rather than training from scratch, we load its pretrained weights and fine-tune it on our water data. ResNet's key trick is **residual connections**: each layer can learn what to *add* to the previous result rather than relearning everything from zero, which makes deeper networks much easier to train.

In this project, ResNet18 acts as the **encoder** — it compresses the input image down into a compact representation that captures what is in the image, at the cost of spatial detail.

### U-Net — recovering the spatial detail
The encoder compresses; the **decoder** has to expand back to full image size to produce a per-pixel mask. The challenge is that compression throws away location information. **U-Net** solves this with **skip connections**: it takes feature maps from early encoder layers (which still have fine spatial detail) and concatenates them directly into the corresponding decoder layers. This lets the network say both *"there is water here"* (from the deep encoder) and *"exactly these pixels"* (from the early encoder). The result is a high-resolution binary mask of the water region.

### Bounding box algorithm
Once the mask is produced, a tight bounding box is drawn around the detected water. A naive approach would take the outermost white pixel in every direction, but stray noise pixels at the edges would make the box too large. Instead, the algorithm uses **percentile trimming**: it collects all white pixel coordinates and crops 4% from each edge using the 4th and 96th percentiles. This makes the box robust to small prediction errors at the boundary while still covering 92% of the detected water region.

## Setup

Place your dataset under `data/` with the following structure:

```
data/
  train/   _annotations.coco.json + images
  valid/   _annotations.coco.json + images
  test/    _annotations.coco.json + images
```

## Usage

**Train:**
```bash
python water_detection_2.py
```
Saves checkpoints to `segmentation_checkpoints/best.pth`.

**Run on an image:**
```bash
python predict.py photo.jpg output.jpg
```

**Run on webcam:**
```bash
python webcam_detect.py
python webcam_detect.py --camera 1
python webcam_detect.py --save result.jpg
```
- `SPACE` — capture and analyse current frame
- `Q` — quit

## Output

- Water region highlighted with a blue overlay
- Bounding box drawn around the detected water
- Level badge in the top-left corner (`level-0` to `level-4`, with confidence %)

## Level scale

| Level | Meaning |
|---|---|
| 0 | Empty |
| 1 | Low |
| 2 | Half |
| 3 | High |
| 4 | Full |
