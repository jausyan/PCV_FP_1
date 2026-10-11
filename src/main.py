import argparse
import cv2
from utils.utils_ import (DetectorConfig, detect_hand, draw_detection, open_camera, read_frame, show)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    args = ap.parse_args()
    cap = open_camera(args.cam)
    cfg = DetectorConfig()

    while True:
        frame = read_frame(cap)
        if frame is None:
            break
        det, mask = detect_hand(frame, cfg)
        draw_detection(frame, det)
        if not show(frame, mask):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
