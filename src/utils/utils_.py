import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"
MAIN_WIN = "controller"
CTRL_WIN = "controls"

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
class SampleState:
    start: tuple = None
    end: tuple = None
    dragging: bool = False
    pending: tuple = None


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


def show(frame: np.ndarray, mask: np.ndarray) -> int:
    cv2.imshow(MAIN_WIN, frame)
    cv2.imshow("mask", mask)
    return cv2.waitKey(1) & 0xFF


def draw_hud(frame: np.ndarray, state: SampleState, msg: str = "") -> None:
    if state.dragging and state.start and state.end:
        cv2.rectangle(frame, state.start, state.end, (255, 200, 0), 2)
    text = "drag=sample hand | s=save | l=load | q=quit"
    cv2.putText(frame, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if msg:
        cv2.putText(frame, msg, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


# ----------------------------------------------------------------- config
def save_config(cfg: DetectorConfig, path: Path = CONFIG_PATH) -> None:
    path.write_text(json.dumps(asdict(cfg), indent=2))


def load_config(path: Path = CONFIG_PATH) -> DetectorConfig:
    try:
        data = json.loads(path.read_text())
        base = asdict(DetectorConfig())
        base.update({k: tuple(v) if isinstance(v, list) else v
                     for k, v in data.items() if k in base})
        return DetectorConfig(**base)
    except (OSError, ValueError, TypeError):
        return DetectorConfig()


# -------------------------------------------------------------- trackbars
_BARS = [
    ("H lo", "hsv_lower", 0, 179), ("H hi", "hsv_upper", 0, 179),
    ("S lo", "hsv_lower", 1, 255), ("S hi", "hsv_upper", 1, 255),
    ("V lo", "hsv_lower", 2, 255), ("V hi", "hsv_upper", 2, 255),
    ("Y lo", "ycrcb_lower", 0, 255), ("Y hi", "ycrcb_upper", 0, 255),
    ("Cr lo", "ycrcb_lower", 1, 255), ("Cr hi", "ycrcb_upper", 1, 255),
    ("Cb lo", "ycrcb_lower", 2, 255), ("Cb hi", "ycrcb_upper", 2, 255),
]


def _noop(_):
    pass


def create_trackbars(cfg: DetectorConfig) -> None:
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTRL_WIN, 420, 520)
    for name, _field, _i, vmax in _BARS:
        cv2.createTrackbar(name, CTRL_WIN, 0, vmax, _noop)
    cv2.createTrackbar("Blur", CTRL_WIN, 0, 15, _noop)
    cv2.createTrackbar("Morph", CTRL_WIN, 0, 31, _noop)
    set_trackbars(cfg)


def set_trackbars(cfg: DetectorConfig) -> None:
    for name, fld, i, _ in _BARS:
        cv2.setTrackbarPos(name, CTRL_WIN, int(getattr(cfg, fld)[i]))
    cv2.setTrackbarPos("Blur", CTRL_WIN, cfg.blur_px)
    cv2.setTrackbarPos("Morph", CTRL_WIN, cfg.morph_kernel_px)


def read_trackbars(cfg: DetectorConfig) -> None:
    vals = {f: list(getattr(cfg, f)) for f in
            ("hsv_lower", "hsv_upper", "ycrcb_lower", "ycrcb_upper")}
    for name, fld, i, _ in _BARS:
        vals[fld][i] = cv2.getTrackbarPos(name, CTRL_WIN)
    for fld, v in vals.items():
        setattr(cfg, fld, tuple(v))
    blur = cv2.getTrackbarPos("Blur", CTRL_WIN)
    cfg.blur_px = blur + 1 if blur > 1 and blur % 2 == 0 else blur
    cfg.morph_kernel_px = cv2.getTrackbarPos("Morph", CTRL_WIN)


# --------------------------------------------------------------- sampling
def setup_sampling(state: SampleState) -> None:
    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.start, state.end, state.dragging = (x, y), (x, y), True
        elif event == cv2.EVENT_MOUSEMOVE and state.dragging:
            state.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state.dragging:
            state.end, state.dragging = (x, y), False
            state.pending = (*state.start, *state.end)

    cv2.setMouseCallback(MAIN_WIN, on_mouse)


def sample_skin(frame: np.ndarray, rect: tuple, cfg: DetectorConfig, k: float = 2.5) -> bool:
    x0, y0, x1, y1 = rect
    x0, x1 = sorted((max(0, x0), min(frame.shape[1], x1)))
    y0, y1 = sorted((max(0, y0), min(frame.shape[0], y1)))
    if (x1 - x0) * (y1 - y0) < 100:
        return False

    patch = cv2.GaussianBlur(frame, (5, 5), 0)[y0:y1, x0:x1]

    def bounds(img, vmax):
        px = img.reshape(-1, 3).astype(np.float32)
        mean, std = px.mean(0), px.std(0)
        lo = np.clip(mean - k * std, 0, vmax)
        hi = np.clip(mean + k * std, 0, vmax)
        return tuple(int(v) for v in lo), tuple(int(v) for v in hi)

    cfg.hsv_lower, cfg.hsv_upper = bounds(
        cv2.cvtColor(patch, cv2.COLOR_BGR2HSV), np.array([179, 255, 255]))
    cfg.ycrcb_lower, cfg.ycrcb_upper = bounds(
        cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb), 255)
    return True
