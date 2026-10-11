from dataclasses import dataclass, field
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
    bbox: tuple  # x, y, w, h
    center: tuple  # cx, cy
    area: float


@dataclass
class HandDetector:
    cfg: DetectorConfig = field(default_factory=DetectorConfig)

    def make_mask(self, frame: np.ndarray) -> np.ndarray:
        c = self.cfg
        if c.blur_px > 1:
            frame = cv2.GaussianBlur(frame, (c.blur_px, c.blur_px), 0)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.bitwise_and(
            cv2.inRange(hsv, np.array(c.hsv_lower), np.array(c.hsv_upper)),
            cv2.inRange(ycrcb, np.array(c.ycrcb_lower), np.array(c.ycrcb_upper)),
        )

        if c.morph_kernel_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (c.morph_kernel_px, c.morph_kernel_px)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)   
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)  
        return mask

    def detect(self, frame: np.ndarray):
        mask = self.make_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask

        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        ratio = area / float(frame.shape[0] * frame.shape[1])
        if ratio < self.cfg.min_area_ratio or ratio > self.cfg.max_area_ratio:
            return None, mask

        x, y, w, h = cv2.boundingRect(best)
        m = cv2.moments(best)
        cx = int(m["m10"] / m["m00"]) if m["m00"] else x + w // 2
        cy = int(m["m01"] / m["m00"]) if m["m00"] else y + h // 2
        return Detection((x, y, w, h), (cx, cy), area), mask
