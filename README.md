# LiftOff-CV V1
by Mohammad Khirz El Jausyan - 5024241009

a simple hand vision controlled drone simulator for beginner

## Features

the Development now Oct 11, I Just added a mirrored webcam feed and uses HSV and YCrCb color thresholds to detect a hand. Gaussian blur and morphological filtering help reduce image noise. When a hand is detected, the app marks its bounding box and center point, while a separate window shows the detection mask. You can select a camera with `--cam` and quit by pressing `q`. Hand movement controls for the 2D drone simulator are planned.

## Run

Install dependencies and start :

```bash
pip install -r requirements.txt
python src/main.py
```

Use `--cam 1` to select another camera. Press `q` to quit.

## Tech stack and approach

- **Python** for the application.
- **OpenCV** for webcam input, image processing, and display.
- **NumPy** for image and threshold operations.
- **Hand detection:** color thresholding in HSV and YCrCb, with blur and morphological filtering. ROI restriction and background rejection are planned to improve detection.
- **Game:** a 2D drone simulator controlled by hand movements; game controls and simulator are planned.
