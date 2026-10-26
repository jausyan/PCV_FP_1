
import json
import math
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pygame


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"
MAIN_WIN = "controller"
CTRL_WIN = "controls"
MASK_WIN = "mask"
BLUR_KERNEL = (5, 5)

GAME_SIZE = (800, 600)
DRONE_R = 28
ACCEL = 1100.0
DRAG = 2.0
MAX_V = 420.0


def _clip(v, lo, hi):
    return max(lo, min(hi, v))



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
    min_solidity: float = 0.45
    max_aspect: float = 3.5


    roi: tuple = (0.35, 0.0, 1.0, 1.0)
    use_tracking: bool = True
    track_pad: float = 0.4
    lost_frames: int = 5


    use_face: bool = True
    face_pad: float = 0.25
    face_every: int = 5
    bg_thresh: int = 25
    bg_frames: int = 30


    smooth_alpha: float = 0.4
    max_jump_px: float = 150.0
    stable_frames: int = 3
    hold_frames: int = 3


    deadzone: float = 0.15
    speed_area_min: float = 0.03
    speed_area_max: float = 0.15


@dataclass
class Detection:
    bbox: tuple
    center: tuple
    area: float


@dataclass
class MouseState:

    sample_start: tuple = None
    sample_end: tuple = None
    sample_dragging: bool = False
    sample_pending: tuple = None
    roi_start: tuple = None
    roi_end: tuple = None
    roi_dragging: bool = False
    roi_pending: tuple = None


@dataclass
class FaceState:
    rects: tuple = ()
    n: int = 0
    cascade: object = None


@dataclass
class BgState:
    model: np.ndarray = None
    acc: np.ndarray = None
    n: int = 0
    capturing: bool = False


@dataclass
class TrackState:
    bbox: tuple = None
    lost: int = 0


@dataclass
class HandFilter:
    smoothed: np.ndarray = None
    last_center: tuple = None
    stable: int = 0
    lost: int = 0


@dataclass
class Control:
    dx: float = 0.0
    dy: float = 0.0
    gain: float = 1.0
    present: bool = False


@dataclass
class Drone:
    x: float = GAME_SIZE[0] / 2
    y: float = GAME_SIZE[1] / 2
    vx: float = 0.0
    vy: float = 0.0
    spin: float = 0.0


@dataclass
class Grabber:
    cap: cv2.VideoCapture
    frame: np.ndarray = None
    frame_id: int = 0
    running: bool = True
    lock: threading.Lock = None
    thread: threading.Thread = None


@dataclass
class Session:

    cfg: DetectorConfig
    mouse: MouseState = field(default_factory=MouseState)
    track: TrackState = field(default_factory=TrackState)
    face: FaceState = field(default_factory=FaceState)
    bg: BgState = field(default_factory=BgState)
    hand_filter: HandFilter = field(default_factory=HandFilter)
    search: tuple = None
    msg: str = ""



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


def start_grabber(cap: cv2.VideoCapture) -> Grabber:

    g = Grabber(cap, lock=threading.Lock())

    def run():
        while g.running:
            frame = read_frame(g.cap)
            if frame is None:
                continue
            with g.lock:
                g.frame, g.frame_id = frame, g.frame_id + 1

    g.thread = threading.Thread(target=run, daemon=True)
    g.thread.start()
    return g


def get_frame(g: Grabber, last_id: int):

    with g.lock:
        if g.frame is None or g.frame_id == last_id:
            return None, last_id
        return g.frame.copy(), g.frame_id


def stop_grabber(g: Grabber) -> None:
    g.running = False
    g.thread.join(timeout=1)



def make_mask(frame: np.ndarray, cfg: DetectorConfig, keep=None) -> np.ndarray:

    if cfg.blur_px > 1:
        frame = cv2.GaussianBlur(frame, (cfg.blur_px, cfg.blur_px), 0)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mask = cv2.bitwise_and(
        cv2.inRange(hsv, np.array(cfg.hsv_lower), np.array(cfg.hsv_upper)),
        cv2.inRange(ycrcb, np.array(cfg.ycrcb_lower), np.array(cfg.ycrcb_upper)),
    )
    if keep is not None:
        mask = cv2.bitwise_and(mask, keep)

    if cfg.morph_kernel_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_kernel_px, cfg.morph_kernel_px))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_hand(mask: np.ndarray, cfg: DetectorConfig, offset=(0, 0), frame_area=None):





    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    ratio = area / float(frame_area or mask.shape[0] * mask.shape[1])
    if not cfg.min_area_ratio <= ratio <= cfg.max_area_ratio:
        return None

    x, y, w, h = cv2.boundingRect(best)
    hull_area = cv2.contourArea(cv2.convexHull(best))
    if hull_area > 0 and area / hull_area < cfg.min_solidity:
        return None
    if max(w, h) / max(1, min(w, h)) > cfg.max_aspect:
        return None

    m = cv2.moments(best)
    cx = int(m["m10"] / m["m00"]) if m["m00"] else x + w // 2
    cy = int(m["m01"] / m["m00"]) if m["m00"] else y + h // 2
    ox, oy = offset
    return Detection((x + ox, y + oy, w, h), (cx + ox, cy + oy), area)


def update_faces(frame: np.ndarray, face: FaceState, cfg: DetectorConfig) -> None:

    if not cfg.use_face:
        face.rects = ()
        return
    if face.cascade is None:
        face.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face.n += 1
    if cfg.face_every > 1 and face.n % cfg.face_every != 1:
        return
    gray = cv2.cvtColor(cv2.resize(frame, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
    found = face.cascade.detectMultiScale(gray, 1.2, 5, minSize=(30, 30))
    face.rects = tuple(tuple(int(v * 2) for v in r) for r in found)


def start_bg_capture(bg: BgState) -> None:
    bg.acc, bg.n, bg.capturing = None, 0, True


def update_bg(frame: np.ndarray, bg: BgState, cfg: DetectorConfig) -> bool:

    if not bg.capturing:
        return False
    blurred = cv2.GaussianBlur(frame, BLUR_KERNEL, 0).astype(np.float32)
    bg.acc = blurred if bg.acc is None else bg.acc + blurred
    bg.n += 1
    if bg.n < cfg.bg_frames:
        return False
    bg.model = (bg.acc / bg.n).astype(np.uint8)
    bg.capturing = False
    return True


def keep_mask(frame: np.ndarray, rect: tuple, cfg: DetectorConfig, face: FaceState, bg: BgState):

    x0, y0, x1, y1 = rect
    crop = frame[y0:y1, x0:x1]
    keep = None
    if bg.model is not None and bg.model.shape == frame.shape:
        diff = cv2.absdiff(cv2.GaussianBlur(crop, BLUR_KERNEL, 0), bg.model[y0:y1, x0:x1])
        keep = (diff.max(axis=2) > cfg.bg_thresh).astype(np.uint8) * 255
    for fx, fy, fw, fh in face.rects:
        pad_x, pad_y = int(fw * cfg.face_pad), int(fh * cfg.face_pad)
        if keep is None:
            keep = np.full(crop.shape[:2], 255, np.uint8)
        cv2.rectangle(keep, (fx - pad_x - x0, fy - pad_y - y0),
                      (fx + fw + pad_x - x0, fy + fh + pad_y - y0), 0, -1)
    return keep


def roi_rect(shape, cfg: DetectorConfig) -> tuple:

    h, w = shape[:2]
    x0, y0, x1, y1 = cfg.roi
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def search_rect(shape, cfg: DetectorConfig, track: TrackState) -> tuple:

    rx0, ry0, rx1, ry1 = roi_rect(shape, cfg)
    if not (cfg.use_tracking and track.bbox):
        return rx0, ry0, rx1, ry1
    x, y, w, h = track.bbox
    pad_x, pad_y = int(w * cfg.track_pad), int(h * cfg.track_pad)
    return (max(rx0, x - pad_x), max(ry0, y - pad_y),
            min(rx1, x + w + pad_x), min(ry1, y + h + pad_y))


def update_track(track: TrackState, det, cfg: DetectorConfig) -> None:
    if det:
        track.bbox, track.lost = det.bbox, 0
        return
    track.lost += 1
    if track.lost >= cfg.lost_frames:
        track.bbox = None


def set_roi(cfg: DetectorConfig, rect: tuple, shape) -> bool:

    h, w = shape[:2]
    x0, y0, x1, y1 = rect
    x0, x1 = sorted((max(0, x0), min(w, x1)))
    y0, y1 = sorted((max(0, y0), min(h, y1)))
    if x1 - x0 < 40 or y1 - y0 < 40:
        return False
    cfg.roi = (x0 / w, y0 / h, x1 / w, y1 / h)
    return True


def detect_hand(frame: np.ndarray, cfg: DetectorConfig, track: TrackState,
                face: FaceState = None, bg: BgState = None):

    x0, y0, x1, y1 = rect = search_rect(frame.shape, cfg, track)
    mask = np.zeros(frame.shape[:2], np.uint8)
    det = None
    if x1 - x0 > 1 and y1 - y0 > 1:
        keep = keep_mask(frame, rect, cfg, face or FaceState(), bg or BgState())
        mask[y0:y1, x0:x1] = make_mask(frame[y0:y1, x0:x1], cfg, keep)
        det = find_hand(mask[y0:y1, x0:x1], cfg, (x0, y0), frame.shape[0] * frame.shape[1])
    update_track(track, det, cfg)
    return det, mask, rect


def filter_hand(hf: HandFilter, det, cfg: DetectorConfig):





    if det is not None:
        raw = np.array([*det.bbox, *det.center, det.area], np.float32)
        jumped = hf.last_center is not None and math.hypot(
            det.center[0] - hf.last_center[0], det.center[1] - hf.last_center[1]) > cfg.max_jump_px
        hf.stable = 1 if (jumped or hf.last_center is None) else hf.stable + 1
        hf.last_center, hf.lost = det.center, 0
        if jumped or hf.smoothed is None:
            hf.smoothed = raw
        else:
            alpha = max(0.05, cfg.smooth_alpha)
            hf.smoothed = alpha * raw + (1 - alpha) * hf.smoothed
    else:
        hf.lost += 1
        if hf.lost > cfg.hold_frames:
            hf.smoothed, hf.last_center, hf.stable = None, None, 0

    if hf.smoothed is None or hf.stable < cfg.stable_frames:
        return None
    x, y, w, h, cx, cy, area = hf.smoothed
    return Detection((int(x), int(y), int(w), int(h)), (int(cx), int(cy)), float(area))



def _deadzone(v: float, dz: float) -> float:

    v = _clip(v, -1.0, 1.0)
    if abs(v) <= dz:
        return 0.0
    return math.copysign((abs(v) - dz) / (1 - dz), v)


def hand_to_control(det, cfg: DetectorConfig, shape) -> Control:

    if det is None:
        return Control()
    h, w = shape[:2]
    x0, y0, x1, y1 = roi_rect(shape, cfg)
    nx = (det.center[0] - (x0 + x1) / 2) / max(1, (x1 - x0) / 2)
    ny = ((y0 + y1) / 2 - det.center[1]) / max(1, (y1 - y0) / 2)
    span = max(1e-6, cfg.speed_area_max - cfg.speed_area_min)
    closeness = _clip((det.area / (w * h) - cfg.speed_area_min) / span, 0.0, 1.0)
    return Control(_deadzone(nx, cfg.deadzone), _deadzone(ny, cfg.deadzone),
                   0.6 + 0.8 * closeness, True)



def init_game():
    pygame.init()
    screen = pygame.display.set_mode(GAME_SIZE)
    pygame.display.set_caption("drone")
    return screen, pygame.time.Clock(), pygame.font.SysFont(None, 24)


def game_running() -> bool:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return False
    return True


def update_drone(drone: Drone, ctrl: Control, dt: float) -> None:
    drone.vx += ctrl.dx * ACCEL * ctrl.gain * dt
    drone.vy -= ctrl.dy * ACCEL * ctrl.gain * dt

    damping = math.exp(-DRAG * dt)
    drone.vx, drone.vy = drone.vx * damping, drone.vy * damping
    top_speed = MAX_V * ctrl.gain
    speed = math.hypot(drone.vx, drone.vy)
    if speed > top_speed:
        drone.vx, drone.vy = drone.vx * top_speed / speed, drone.vy * top_speed / speed

    drone.x += drone.vx * dt
    drone.y += drone.vy * dt
    w, h = GAME_SIZE
    if not DRONE_R <= drone.x <= w - DRONE_R:
        drone.x, drone.vx = _clip(drone.x, DRONE_R, w - DRONE_R), 0.0
    if not DRONE_R <= drone.y <= h - DRONE_R:
        drone.y, drone.vy = _clip(drone.y, DRONE_R, h - DRONE_R), 0.0
    drone.spin += dt * (18 + math.hypot(drone.vx, drone.vy) * 0.05)


def _draw_bar(screen, font, label, value, y):

    screen.blit(font.render(label, True, (200, 200, 200)), (12, y))
    pygame.draw.rect(screen, (70, 70, 80), (70, y + 2, 120, 12))
    if value >= 0:
        fill = (130, y + 2, int(60 * value), 12)
    else:
        fill = (130 + int(60 * value), y + 2, int(-60 * value), 12)
    pygame.draw.rect(screen, (90, 200, 120), fill)


def _draw_grid(screen):
    screen.fill((24, 28, 38))
    for gx in range(0, GAME_SIZE[0], 50):
        pygame.draw.line(screen, (34, 40, 54), (gx, 0), (gx, GAME_SIZE[1]))
    for gy in range(0, GAME_SIZE[1], 50):
        pygame.draw.line(screen, (34, 40, 54), (0, gy), (GAME_SIZE[0], gy))


def _draw_drone(screen, drone: Drone, present: bool):
    pygame.draw.ellipse(screen, (16, 18, 24), (drone.x - 26, drone.y + 24, 52, 12))
    tilt = math.radians(_clip(drone.vx / MAX_V, -1.0, 1.0) * 20)
    cos_t, sin_t = math.cos(tilt), math.sin(tilt)

    def rotated(px, py):
        return drone.x + px * cos_t - py * sin_t, drone.y + px * sin_t + py * cos_t

    for arm_x, arm_y in ((-24, -10), (24, -10), (-24, 10), (24, 10)):
        pygame.draw.line(screen, (150, 150, 160), rotated(0, 0), rotated(arm_x, arm_y), 4)
        rx, ry = rotated(arm_x, arm_y)
        for blade in range(2):
            a = drone.spin + blade * math.pi / 2
            pygame.draw.line(screen, (230, 230, 240),
                             (rx - 11 * math.cos(a), ry - 4 * math.sin(a)),
                             (rx + 11 * math.cos(a), ry + 4 * math.sin(a)), 2)
    body = (80, 170, 255) if present else (150, 80, 80)
    pygame.draw.circle(screen, body, (int(drone.x), int(drone.y)), 11)


def draw_game(screen, font, drone: Drone, ctrl: Control) -> None:
    _draw_grid(screen)
    _draw_drone(screen, drone, ctrl.present)
    _draw_bar(screen, font, "x", ctrl.dx, 12)
    _draw_bar(screen, font, "y", ctrl.dy, 32)
    screen.blit(font.render(f"speed x{ctrl.gain:.2f}", True, (200, 200, 200)), (12, 54))
    if not ctrl.present:
        screen.blit(font.render("NO HAND", True, (240, 90, 90)), (12, 78))
    pygame.display.flip()



def save_config(cfg: DetectorConfig, path: Path = CONFIG_PATH) -> None:
    path.write_text(json.dumps(asdict(cfg), indent=2))


def load_config(path: Path = CONFIG_PATH) -> DetectorConfig:

    try:
        data = json.loads(path.read_text())
        merged = asdict(DetectorConfig())
        merged.update({k: tuple(v) if isinstance(v, list) else v
                       for k, v in data.items() if k in merged})
        return DetectorConfig(**merged)
    except (OSError, ValueError, TypeError):
        return DetectorConfig()




_TRACKBARS = [
    ("H lo", "hsv_lower", 0, 179), ("H hi", "hsv_upper", 0, 179),
    ("S lo", "hsv_lower", 1, 255), ("S hi", "hsv_upper", 1, 255),
    ("V lo", "hsv_lower", 2, 255), ("V hi", "hsv_upper", 2, 255),
    ("Y lo", "ycrcb_lower", 0, 255), ("Y hi", "ycrcb_upper", 0, 255),
    ("Cr lo", "ycrcb_lower", 1, 255), ("Cr hi", "ycrcb_upper", 1, 255),
    ("Cb lo", "ycrcb_lower", 2, 255), ("Cb hi", "ycrcb_upper", 2, 255),
]
_RANGE_FIELDS = ("hsv_lower", "hsv_upper", "ycrcb_lower", "ycrcb_upper")


def create_trackbars(cfg: DetectorConfig) -> None:
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTRL_WIN, 420, 640)
    for name, _, _, vmax in _TRACKBARS:
        cv2.createTrackbar(name, CTRL_WIN, 0, vmax, lambda _: None)
    for name, vmax in (("Blur", 15), ("Morph", 31), ("BgDiff", 100), ("Solid%", 100), ("Smooth%", 100)):
        cv2.createTrackbar(name, CTRL_WIN, 0, vmax, lambda _: None)
    set_trackbars(cfg)


def set_trackbars(cfg: DetectorConfig) -> None:

    for name, fld, i, _ in _TRACKBARS:
        cv2.setTrackbarPos(name, CTRL_WIN, int(getattr(cfg, fld)[i]))
    cv2.setTrackbarPos("Blur", CTRL_WIN, cfg.blur_px)
    cv2.setTrackbarPos("Morph", CTRL_WIN, cfg.morph_kernel_px)
    cv2.setTrackbarPos("BgDiff", CTRL_WIN, cfg.bg_thresh)
    cv2.setTrackbarPos("Solid%", CTRL_WIN, int(cfg.min_solidity * 100))
    cv2.setTrackbarPos("Smooth%", CTRL_WIN, int(cfg.smooth_alpha * 100))


def read_trackbars(cfg: DetectorConfig) -> None:

    ranges = {f: list(getattr(cfg, f)) for f in _RANGE_FIELDS}
    for name, fld, i, _ in _TRACKBARS:
        ranges[fld][i] = cv2.getTrackbarPos(name, CTRL_WIN)
    for fld, values in ranges.items():
        setattr(cfg, fld, tuple(values))

    blur = cv2.getTrackbarPos("Blur", CTRL_WIN)
    cfg.blur_px = blur + 1 if blur > 1 and blur % 2 == 0 else blur
    cfg.morph_kernel_px = cv2.getTrackbarPos("Morph", CTRL_WIN)
    cfg.bg_thresh = cv2.getTrackbarPos("BgDiff", CTRL_WIN)
    cfg.min_solidity = cv2.getTrackbarPos("Solid%", CTRL_WIN) / 100
    cfg.smooth_alpha = max(5, cv2.getTrackbarPos("Smooth%", CTRL_WIN)) / 100



def setup_mouse(mouse: MouseState) -> None:
    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse.sample_start, mouse.sample_end, mouse.sample_dragging = (x, y), (x, y), True
        elif event == cv2.EVENT_MOUSEMOVE and mouse.sample_dragging:
            mouse.sample_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and mouse.sample_dragging:
            mouse.sample_end, mouse.sample_dragging = (x, y), False
            mouse.sample_pending = (*mouse.sample_start, *mouse.sample_end)
        elif event == cv2.EVENT_RBUTTONDOWN:
            mouse.roi_start, mouse.roi_end, mouse.roi_dragging = (x, y), (x, y), True
        elif event == cv2.EVENT_MOUSEMOVE and mouse.roi_dragging:
            mouse.roi_end = (x, y)
        elif event == cv2.EVENT_RBUTTONUP and mouse.roi_dragging:
            mouse.roi_end, mouse.roi_dragging = (x, y), False
            mouse.roi_pending = (*mouse.roi_start, *mouse.roi_end)

    cv2.setMouseCallback(MAIN_WIN, on_mouse)


def sample_skin(frame: np.ndarray, rect: tuple, cfg: DetectorConfig, k: float = 2.5) -> bool:

    x0, y0, x1, y1 = rect
    x0, x1 = sorted((max(0, x0), min(frame.shape[1], x1)))
    y0, y1 = sorted((max(0, y0), min(frame.shape[0], y1)))
    if (x1 - x0) * (y1 - y0) < 100:
        return False

    patch = cv2.GaussianBlur(frame, BLUR_KERNEL, 0)[y0:y1, x0:x1]

    def bounds(img, vmax):
        pixels = img.reshape(-1, 3).astype(np.float32)
        mean, std = pixels.mean(0), pixels.std(0)
        lower = np.clip(mean - k * std, 0, vmax)
        upper = np.clip(mean + k * std, 0, vmax)
        return tuple(int(v) for v in lower), tuple(int(v) for v in upper)

    cfg.hsv_lower, cfg.hsv_upper = bounds(
        cv2.cvtColor(patch, cv2.COLOR_BGR2HSV), np.array([179, 255, 255]))
    cfg.ycrcb_lower, cfg.ycrcb_upper = bounds(cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb), 255)
    return True



def draw_rois(frame: np.ndarray, cfg: DetectorConfig, search: tuple) -> None:

    x0, y0, x1, y1 = roi_rect(frame.shape, cfg)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 120, 0), 1)
    if search != (x0, y0, x1, y1):
        cv2.rectangle(frame, search[:2], search[2:], (0, 255, 255), 1)


def draw_faces(frame: np.ndarray, face: FaceState) -> None:
    for x, y, w, h in face.rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 1)


def draw_detection(frame: np.ndarray, det) -> None:
    if det is None:
        return
    x, y, w, h = det.bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(frame, det.center, 5, (0, 0, 255), -1)


def draw_status(frame: np.ndarray, hf: HandFilter, cfg: DetectorConfig) -> None:
    if hf.smoothed is None:
        text, color = "hand: none", (0, 0, 255)
    elif hf.lost > 0:
        text, color = f"hand: holding ({hf.lost}/{cfg.hold_frames})", (0, 165, 255)
    elif hf.stable < cfg.stable_frames:
        text, color = f"hand: confirming ({hf.stable}/{cfg.stable_frames})", (0, 255, 255)
    else:
        text, color = "hand: stable", (0, 255, 0)
    cv2.putText(frame, text, (8, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_hud(frame: np.ndarray, mouse: MouseState, msg: str = "") -> None:
    if mouse.sample_dragging and mouse.sample_start and mouse.sample_end:
        cv2.rectangle(frame, mouse.sample_start, mouse.sample_end, (255, 200, 0), 2)
    if mouse.roi_dragging and mouse.roi_start and mouse.roi_end:
        cv2.rectangle(frame, mouse.roi_start, mouse.roi_end, (255, 120, 0), 2)
    lines = ["L-drag=sample | R-drag=ROI | r=reset ROI | t=tracking | s/l=save/load | q=quit",
             "b=capture bg | x=clear bg | f=face exclusion"]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if msg:
        cv2.putText(frame, msg, (8, 20 + 18 * len(lines)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def show(frame: np.ndarray, mask: np.ndarray) -> int:

    cv2.imshow(MAIN_WIN, frame)
    cv2.imshow(MASK_WIN, mask)
    return cv2.waitKey(1) & 0xFF



def apply_mouse_requests(frame: np.ndarray, s: Session) -> None:

    if s.mouse.sample_pending:
        ok = sample_skin(frame, s.mouse.sample_pending, s.cfg)
        s.msg = "sampled hand color" if ok else "sample too small"
        s.mouse.sample_pending = None
        if ok:
            set_trackbars(s.cfg)
    if s.mouse.roi_pending:
        ok = set_roi(s.cfg, s.mouse.roi_pending, frame.shape)
        s.msg = "ROI set" if ok else "ROI too small"
        s.mouse.roi_pending = None
        s.track.bbox = None


def process_frame(frame: np.ndarray, s: Session):

    read_trackbars(s.cfg)
    if s.bg.capturing:
        if update_bg(frame, s.bg, s.cfg):
            s.msg = "background captured"
        else:
            s.msg = f"capturing background {s.bg.n}/{s.cfg.bg_frames} - keep hand out"
    update_faces(frame, s.face, s.cfg)

    det, mask, s.search = detect_hand(frame, s.cfg, s.track, s.face, s.bg)
    return filter_hand(s.hand_filter, det, s.cfg), mask


def draw_overlays(frame: np.ndarray, s: Session, hand) -> None:
    draw_rois(frame, s.cfg, s.search)
    draw_faces(frame, s.face)
    draw_detection(frame, hand)
    draw_status(frame, s.hand_filter, s.cfg)
    draw_hud(frame, s.mouse, s.msg)


def handle_key(key: int, s: Session) -> bool:

    if key == ord("q"):
        return False
    if key == ord("s"):
        save_config(s.cfg)
        s.msg = "saved config.json"
    elif key == ord("l"):
        s.cfg = load_config()
        set_trackbars(s.cfg)
        s.track.bbox = None
        s.msg = "loaded config.json"
    elif key == ord("r"):
        s.cfg.roi = DetectorConfig().roi
        s.track.bbox = None
        s.msg = "ROI reset"
    elif key == ord("b"):
        start_bg_capture(s.bg)
        s.track.bbox = None
    elif key == ord("x"):
        s.bg.model = None
        s.msg = "background cleared"
    elif key == ord("f"):
        s.cfg.use_face = not s.cfg.use_face
        s.msg = f"face exclusion {'on' if s.cfg.use_face else 'off'}"
    elif key == ord("t"):
        s.cfg.use_tracking = not s.cfg.use_tracking
        s.track.bbox = None
        s.msg = f"tracking {'on' if s.cfg.use_tracking else 'off'}"
    return True
