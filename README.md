# Real-time Privacy Virtual Camera (Shadow Cam)

A real-time Python application that creates a virtual webcam system device and automatically censors the user's silhouette into a clean black shadow using deep learning matting.

## How It Works

1. **Camera Ingestion**: Captures live video stream from your physical camera using OpenCV.
2. **Neural Silhouette Segmentation**: Passes frames through a deep learning model (**Robust Video Matting / RVM**) running on ONNX Runtime. It detects the continuous silhouette of the person (including face, hands, clothes, and gestures) in real-time.
3. **Anti-Aliased Black Fill**: Uses a soft Gaussian alpha mask to blend the person's contour with deep black pixels while keeping the original background visible.
4. **Virtual Camera Output**: Streams the processed video frames directly into an **OBS Virtual Camera** system device via `pyvirtualcam`, making it available to Zoom, Discord, Telegram, Google Meet, or browsers.

## Requirements

- Python 3.10+
- [OBS Studio](https://obsproject.com/) (installed to provide the OBS Virtual Camera system driver)
- A webcam

## Installation

```bash
git clone https://github.com/HumanizeLegendary/shadow-cam.git
cd shadow-cam
pip install -r requirements.txt
```

## Usage

Simply run:

```bash
python shadow_cam.py
```

On first launch, the script will automatically download the lightweight neural ONNX weights. Then:
1. Open any video app (Telegram, Discord, Zoom, etc.).
2. In camera settings, select **OBS Virtual Camera**.
