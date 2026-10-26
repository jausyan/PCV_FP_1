import json
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
import cv2
import numpy as np
import pygame

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"
MAIN_WIN = "controller"
CTRL_WIN = "controls"
GAME_SIZE = (800, 600)
ACCEL, DRAG, MAX_V, DRONE_R = 1100.0, 2.0, 420.0, 28

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
    roi: tuple = (0.35, 0.0, 1.0, 1.0)
    use_tracking: bool = True
    track_pad: float = 0.4
    lost_frames: int = 5
    use_face: bool = True
    face_pad: float = 0.25
    face_every: int = 5
    bg_thresh: int = 25
    bg_frames: int = 30
    min_solidity: float = 0.45
    max_aspect: float = 3.5
    smooth_alpha: float = 0.4
    max_jump_px: float = 150.0
    stable_frames: int = 3
    hold_frames: int = 3
    deadzone: float = 0.15
    area_min_ratio: float = 0.03
    area_max_ratio: float = 0.15


@dataclass
class SampleState:
    start: tuple = None
    end: tuple = None
    dragging: bool = False
    pending: tuple = None
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
    sm: np.ndarray = None  # smoothed [x, y, w, h, cx, cy, area]
    last_center: tuple = None
    stable: int = 0
    lost: int = 0


@dataclass
class Control:
    dx: float = 0.0    # -1 left .. 1 right
    dy: float = 0.0    # -1 down .. 1 up
    gain: float = 1.0  # speed multiplier from hand size
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
    fid: int = 0
    running: bool = True
    lock: threading.Lock = None
    thread: threading.Thread = None


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
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_kernel_px, cfg.morph_kernel_px)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)   
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)  
    return mask


def find_hand(mask: np.ndarray, cfg: DetectorConfig, offset=(0, 0), frame_area=None):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    ratio = area / float(frame_area or mask.shape[0] * mask.shape[1])
    if ratio < cfg.min_area_ratio or ratio > cfg.max_area_ratio:
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
    """Haar face detection every cfg.face_every frames on a half-size gray image."""
    if not cfg.use_face:
        face.rects = ()
        return
    if face.cascade is None:
        face.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face.n += 1
    if face.n % cfg.face_every != 1 and cfg.face_every > 1:
        return
    gray = cv2.cvtColor(cv2.resize(frame, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
    found = face.cascade.detectMultiScale(gray, 1.2, 5, minSize=(30, 30))
    face.rects = tuple(tuple(int(v * 2) for v in r) for r in found)


def start_bg_capture(bg: BgState) -> None:
    bg.acc, bg.n, bg.capturing = None, 0, True


def update_bg(frame: np.ndarray, bg: BgState, cfg: DetectorConfig) -> bool:
    """Accumulate frames while capturing. Returns True when the model is finished."""
    if not bg.capturing:
        return False
    f = cv2.GaussianBlur(frame, (5, 5), 0).astype(np.float32)
    bg.acc = f if bg.acc is None else bg.acc + f
    bg.n += 1
    if bg.n < cfg.bg_frames:
        return False
    bg.model = (bg.acc / bg.n).astype(np.uint8)
    bg.capturing = False
    return True


def keep_mask(frame: np.ndarray, rect: tuple, cfg: DetectorConfig, face: FaceState, bg: BgState):
    """255 where a pixel may be hand: differs from the background and is outside faces."""
    x0, y0, x1, y1 = rect
    crop = frame[y0:y1, x0:x1]
    keep = None
    if bg.model is not None and bg.model.shape == frame.shape:
        diff = cv2.absdiff(cv2.GaussianBlur(crop, (5, 5), 0), bg.model[y0:y1, x0:x1])
        keep = (diff.max(axis=2) > cfg.bg_thresh).astype(np.uint8) * 255
    for fx, fy, fw, fh in face.rects:
        px, py = int(fw * cfg.face_pad), int(fh * cfg.face_pad)
        if keep is None:
            keep = np.full(crop.shape[:2], 255, np.uint8)
        cv2.rectangle(keep, (fx - px - x0, fy - py - y0),
                      (fx + fw + px - x0, fy + fh + py - y0), 0, -1)
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
    px, py = int(w * cfg.track_pad), int(h * cfg.track_pad)
    return (max(rx0, x - px), max(ry0, y - py),
            min(rx1, x + w + px), min(ry1, y + h + py))


def update_track(track: TrackState, det, cfg: DetectorConfig) -> None:
    if det:
        track.bbox, track.lost = det.bbox, 0
    else:
        track.lost += 1
        if track.lost >= cfg.lost_frames:
            track.bbox = None


def detect_hand(frame: np.ndarray, cfg: DetectorConfig, track: TrackState,
                face: FaceState = None, bg: BgState = None):
    x0, y0, x1, y1 = rect = search_rect(frame.shape, cfg, track)
    mask = np.zeros(frame.shape[:2], np.uint8)
    det = None
    if x1 - x0 > 1 and y1 - y0 > 1:
        keep = keep_mask(frame, rect, cfg, face or FaceState(), bg or BgState())
        mask[y0:y1, x0:x1] = make_mask(frame[y0:y1, x0:x1], cfg, keep)
        det = find_hand(mask[y0:y1, x0:x1], cfg, (x0, y0),
                        frame.shape[0] * frame.shape[1])
    update_track(track, det, cfg)
    return det, mask, rect


def filter_hand(hf: HandFilter, det, cfg: DetectorConfig):
    """Jump rejection + stable-frame gating + EMA smoothing + short hold on loss.

    Returns the smoothed Detection, or None until the hand is stable / once it is lost.
    """
    if det is not None:
        raw = np.array([*det.bbox, *det.center, det.area], np.float32)
        jumped = (hf.last_center is not None and
                  np.hypot(det.center[0] - hf.last_center[0],
                           det.center[1] - hf.last_center[1]) > cfg.max_jump_px)
        hf.stable = 1 if (jumped or hf.last_center is None) else hf.stable + 1
        hf.last_center, hf.lost = det.center, 0
        if jumped or hf.sm is None:
            hf.sm = raw
        else:
            a = max(0.05, cfg.smooth_alpha)
            hf.sm = a * raw + (1 - a) * hf.sm
    else:
        hf.lost += 1
        if hf.lost > cfg.hold_frames:
            hf.sm, hf.last_center, hf.stable = None, None, 0

    if hf.sm is None or hf.stable < cfg.stable_frames:
        return None
    x, y, w, h, cx, cy, area = hf.sm
    return Detection((int(x), int(y), int(w), int(h)), (int(cx), int(cy)), float(area))


def set_roi(cfg: DetectorConfig, rect: tuple, shape) -> bool:
    h, w = shape[:2]
    x0, y0, x1, y1 = rect
    x0, x1 = sorted((max(0, x0), min(w, x1)))
    y0, y1 = sorted((max(0, y0), min(h, y1)))
    if (x1 - x0) < 40 or (y1 - y0) < 40:
        return False
    cfg.roi = (x0 / w, y0 / h, x1 / w, y1 / h)
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
    if hf.sm is None:
        text, color = "hand: none", (0, 0, 255)
    elif hf.lost > 0:
        text, color = f"hand: holding ({hf.lost}/{cfg.hold_frames})", (0, 165, 255)
    elif hf.stable < cfg.stable_frames:
        text, color = f"hand: confirming ({hf.stable}/{cfg.stable_frames})", (0, 255, 255)
    else:
        text, color = "hand: stable", (0, 255, 0)
    cv2.putText(frame, text, (8, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def show(frame: np.ndarray, mask: np.ndarray) -> int:
    cv2.imshow(MAIN_WIN, frame)
    cv2.imshow("mask", mask)
    return cv2.waitKey(1) & 0xFF


def draw_hud(frame: np.ndarray, state: SampleState, msg: str = "") -> None:
    if state.dragging and state.start and state.end:
        cv2.rectangle(frame, state.start, state.end, (255, 200, 0), 2)
    if state.roi_dragging and state.roi_start and state.roi_end:
        cv2.rectangle(frame, state.roi_start, state.roi_end, (255, 120, 0), 2)
    lines = ["L-drag=sample | R-drag=ROI | r=reset ROI | t=tracking | s/l=save/load | q=quit",
             "b=capture bg | x=clear bg | f=face exclusion"]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if msg:
        cv2.putText(frame, msg, (8, 20 + 18 * len(lines)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

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
    cv2.resizeWindow(CTRL_WIN, 420, 640)
    for name, _field, _i, vmax in _BARS:
        cv2.createTrackbar(name, CTRL_WIN, 0, vmax, _noop)
    cv2.createTrackbar("Blur", CTRL_WIN, 0, 15, _noop)
    cv2.createTrackbar("Morph", CTRL_WIN, 0, 31, _noop)
    cv2.createTrackbar("BgDiff", CTRL_WIN, 0, 100, _noop)
    cv2.createTrackbar("Solid%", CTRL_WIN, 0, 100, _noop)
    cv2.createTrackbar("Smooth%", CTRL_WIN, 5, 100, _noop)
    set_trackbars(cfg)


def set_trackbars(cfg: DetectorConfig) -> None:
    for name, fld, i, _ in _BARS:
        cv2.setTrackbarPos(name, CTRL_WIN, int(getattr(cfg, fld)[i]))
    cv2.setTrackbarPos("Blur", CTRL_WIN, cfg.blur_px)
    cv2.setTrackbarPos("Morph", CTRL_WIN, cfg.morph_kernel_px)
    cv2.setTrackbarPos("BgDiff", CTRL_WIN, cfg.bg_thresh)
    cv2.setTrackbarPos("Solid%", CTRL_WIN, int(cfg.min_solidity * 100))
    cv2.setTrackbarPos("Smooth%", CTRL_WIN, int(cfg.smooth_alpha * 100))


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
    cfg.bg_thresh = cv2.getTrackbarPos("BgDiff", CTRL_WIN)
    cfg.min_solidity = cv2.getTrackbarPos("Solid%", CTRL_WIN) / 100
    cfg.smooth_alpha = max(5, cv2.getTrackbarPos("Smooth%", CTRL_WIN)) / 100

# ------------------------------------------------------------ camera thread
def start_grabber(cap: cv2.VideoCapture) -> Grabber:
    g = Grabber(cap, lock=threading.Lock())

    def run():
        while g.running:
            frame = read_frame(g.cap)
            if frame is None:
                continue
            with g.lock:
                g.frame, g.fid = frame, g.fid + 1

    g.thread = threading.Thread(target=run, daemon=True)
    g.thread.start()
    return g


def get_frame(g: Grabber, last_id: int):
    """Newest frame (a copy) and its id, or (None, last_id) if nothing new."""
    with g.lock:
        if g.frame is None or g.fid == last_id:
            return None, last_id
        return g.frame.copy(), g.fid


def stop_grabber(g: Grabber) -> None:
    g.running = False
    g.thread.join(timeout=1)


# ------------------------------------------------------------ hand -> drone
def _deadzone(v: float, dz: float) -> float:
    v = max(-1.0, min(1.0, v))
    if abs(v) <= dz:
        return 0.0
    return math.copysign((abs(v) - dz) / (1 - dz), v)


def hand_to_control(det, cfg: DetectorConfig, shape) -> Control:
    """Offset from the ROI center -> dx/dy, hand size -> speed gain. No hand -> neutral."""
    if det is None:
        return Control()
    h, w = shape[:2]
    x0, y0, x1, y1 = roi_rect(shape, cfg)
    nx = (det.center[0] - (x0 + x1) / 2) / max(1, (x1 - x0) / 2)
    ny = ((y0 + y1) / 2 - det.center[1]) / max(1, (y1 - y0) / 2)
    span = max(1e-6, cfg.area_max_ratio - cfg.area_min_ratio)
    t = max(0.0, min(1.0, (det.area / (w * h) - cfg.area_min_ratio) / span))
    return Control(_deadzone(nx, cfg.deadzone), _deadzone(ny, cfg.deadzone), 0.6 + 0.8 * t, True)


# --------------------------------------------------------------- game
def init_game():
    pygame.init()
    screen = pygame.display.set_mode(GAME_SIZE)
    pygame.display.set_caption("drone")
    return screen, pygame.time.Clock(), pygame.font.SysFont(None, 24)


def game_running() -> bool:
    for e in pygame.event.get():
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
            return False
    return True


def update_drone(d: Drone, c: Control, dt: float) -> None:
    d.vx += c.dx * ACCEL * c.gain * dt
    d.vy -= c.dy * ACCEL * c.gain * dt  # screen y points down
    damp = math.exp(-DRAG * dt)
    d.vx, d.vy = d.vx * damp, d.vy * damp
    vmax = MAX_V * c.gain
    speed = math.hypot(d.vx, d.vy)
    if speed > vmax:
        d.vx, d.vy = d.vx * vmax / speed, d.vy * vmax / speed
    d.x += d.vx * dt
    d.y += d.vy * dt
    w, h = GAME_SIZE
    if not DRONE_R <= d.x <= w - DRONE_R:
        d.x, d.vx = min(max(d.x, DRONE_R), w - DRONE_R), 0.0
    if not DRONE_R <= d.y <= h - DRONE_R:
        d.y, d.vy = min(max(d.y, DRONE_R), h - DRONE_R), 0.0
    d.spin += dt * (18 + math.hypot(d.vx, d.vy) * 0.05)


def _bar(screen, font, label, v, y):
    screen.blit(font.render(label, True, (200, 200, 200)), (12, y))
    pygame.draw.rect(screen, (70, 70, 80), (70, y + 2, 120, 12))
    bar = (130, y + 2, int(60 * v), 12) if v >= 0 else (130 + int(60 * v), y + 2, int(-60 * v), 12)
    pygame.draw.rect(screen, (90, 200, 120), bar)


def draw_game(screen, font, d: Drone, c: Control) -> None:
    screen.fill((24, 28, 38))
    for gx in range(0, GAME_SIZE[0], 50):
        pygame.draw.line(screen, (34, 40, 54), (gx, 0), (gx, GAME_SIZE[1]))
    for gy in range(0, GAME_SIZE[1], 50):
        pygame.draw.line(screen, (34, 40, 54), (0, gy), (GAME_SIZE[0], gy))

    pygame.draw.ellipse(screen, (16, 18, 24), (d.x - 26, d.y + 24, 52, 12))  # shadow
    tilt = math.radians(max(-1.0, min(1.0, d.vx / MAX_V)) * 20)
    cs, sn = math.cos(tilt), math.sin(tilt)

    def pt(px, py):
        return d.x + px * cs - py * sn, d.y + px * sn + py * cs

    for rx, ry in ((-24, -10), (24, -10), (-24, 10), (24, 10)):
        pygame.draw.line(screen, (150, 150, 160), pt(0, 0), pt(rx, ry), 4)
        cx, cy = pt(rx, ry)
        for k in range(2):  # spinning blades
            a = d.spin + k * math.pi / 2
            pygame.draw.line(screen, (230, 230, 240),
                             (cx - 11 * math.cos(a), cy - 4 * math.sin(a)),
                             (cx + 11 * math.cos(a), cy + 4 * math.sin(a)), 2)
    pygame.draw.circle(screen, (80, 170, 255) if c.present else (150, 80, 80), (int(d.x), int(d.y)), 11)

    _bar(screen, font, "x", c.dx, 12)
    _bar(screen, font, "y", c.dy, 32)
    screen.blit(font.render(f"speed x{c.gain:.2f}", True, (200, 200, 200)), (12, 54))
    if not c.present:
        screen.blit(font.render("NO HAND", True, (240, 90, 90)), (12, 78))
    pygame.display.flip()


def setup_sampling(state: SampleState) -> None:
    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.start, state.end, state.dragging = (x, y), (x, y), True
        elif event == cv2.EVENT_MOUSEMOVE and state.dragging:
            state.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state.dragging:
            state.end, state.dragging = (x, y), False
            state.pending = (*state.start, *state.end)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state.roi_start, state.roi_end, state.roi_dragging = (x, y), (x, y), True
        elif event == cv2.EVENT_MOUSEMOVE and state.roi_dragging:
            state.roi_end = (x, y)
        elif event == cv2.EVENT_RBUTTONUP and state.roi_dragging:
            state.roi_end, state.roi_dragging = (x, y), False
            state.roi_pending = (*state.roi_start, *state.roi_end)

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
