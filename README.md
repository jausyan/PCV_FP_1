# LiftOff-CV V1
by Mohammad Khirz El Jausyan - 5024241009

a simple hand vision controlled drone simulator for beginner

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
