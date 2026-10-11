from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class DetectorConfig:
    hsv_lower: tuple = (0, 40, 60)
    hsv_upper: tuple = (25, 255, 255)
    ycrcb_lower: tuple = (0, 133, 77)
    ycrcb_upper: tuple = (255, 173, 127)
    blur_px: int = 5
    morph_kernel_px: int = 7
    min_area_ratio: float = 0.01
    max_area_ratio: float = 0.8


@dataclass
class Detection:
    bbox: tuple  
    center: tuple  
    area: float

def open_camera(index: int = 0, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {index}")
    return cap


def read_frame(cap: cv2.VideoCapture):
    ok, frame = cap.read()
    return cv2.flip(frame, 1) if ok else None

def make_mask(frame: np.ndarray, cfg: DetectorConfig) -> np.ndarray:
    if cfg.blur_px > 1:
        frame = cv2.GaussianBlur(frame, (cfg.blur_px, cfg.blur_px), 0)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mask = cv2.bitwise_and(
        cv2.inRange(hsv, np.array(cfg.hsv_lower), np.array(cfg.hsv_upper)),
        cv2.inRange(ycrcb, np.array(cfg.ycrcb_lower), np.array(cfg.ycrcb_upper)),
    )

    if cfg.morph_kernel_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_kernel_px, cfg.morph_kernel_px)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)   
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)  
    return mask


def find_hand(mask: np.ndarray, cfg: DetectorConfig):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    ratio = area / float(mask.shape[0] * mask.shape[1])
    if ratio < cfg.min_area_ratio or ratio > cfg.max_area_ratio:
        return None

    x, y, w, h = cv2.boundingRect(best)
    m = cv2.moments(best)
    cx = int(m["m10"] / m["m00"]) if m["m00"] else x + w // 2
    cy = int(m["m01"] / m["m00"]) if m["m00"] else y + h // 2
    return Detection((x, y, w, h), (cx, cy), area)


def detect_hand(frame: np.ndarray, cfg: DetectorConfig):
    mask = make_mask(frame, cfg)
    return find_hand(mask, cfg), mask

def draw_detection(frame: np.ndarray, det) -> None:
    if det is None:
        return
    x, y, w, h = det.bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(frame, det.center, 5, (0, 0, 255), -1)


def show(frame: np.ndarray, mask: np.ndarray) -> bool:
    """Show both windows. Returns False when the user presses q."""
    cv2.imshow("controller", frame)
    cv2.imshow("mask", mask)
    return (cv2.waitKey(1) & 0xFF) != ord("q")
