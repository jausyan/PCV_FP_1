import argparse
import cv2
from utils.utils_ import (MAIN_WIN, BgState, DetectorConfig, FaceState, SampleState, TrackState, create_trackbars, detect_hand, draw_detection, draw_faces, draw_hud, draw_rois, load_config, open_camera, read_frame, read_trackbars, sample_skin, save_config, set_roi, set_trackbars, setup_sampling, show, start_bg_capture, update_bg, update_faces)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    args = ap.parse_args()
    cap = open_camera(args.cam)
    cfg = load_config()
    state = SampleState()
    track = TrackState()
    face = FaceState()
    bg = BgState()
    msg = ""
    cv2.namedWindow(MAIN_WIN)
    create_trackbars(cfg)
    setup_sampling(state)

    while True:
        frame = read_frame(cap)
        if frame is None:
            break

        if state.pending:
            ok = sample_skin(frame, state.pending, cfg)
            msg = "sampled hand color" if ok else "sample too small"
            state.pending = None
            if ok:
                set_trackbars(cfg)
        if state.roi_pending:
            ok = set_roi(cfg, state.roi_pending, frame.shape)
            msg = "ROI set" if ok else "ROI too small"
            state.roi_pending = None
            track.bbox = None
        read_trackbars(cfg)

        if bg.capturing:
            if update_bg(frame, bg, cfg):
                msg = "background captured"
            else:
                msg = f"capturing background {bg.n}/{cfg.bg_frames} - keep hand out"
        update_faces(frame, face, cfg)

        det, mask, search = detect_hand(frame, cfg, track, face, bg)
        draw_rois(frame, cfg, search)
        draw_faces(frame, face)
        draw_detection(frame, det)
        draw_hud(frame, state, msg)

        key = show(frame, mask)
        if key == ord("q"):
            break
        if key == ord("s"):
            save_config(cfg)
            msg = "saved config.json"
        elif key == ord("l"):
            cfg = load_config()
            set_trackbars(cfg)
            track.bbox = None
            msg = "loaded config.json"
        elif key == ord("r"):
            cfg.roi = DetectorConfig().roi
            track.bbox = None
            msg = "ROI reset"
        elif key == ord("b"):
            start_bg_capture(bg)
            track.bbox = None
        elif key == ord("x"):
            bg.model = None
            msg = "background cleared"
        elif key == ord("f"):
            cfg.use_face = not cfg.use_face
            msg = f"face exclusion {'on' if cfg.use_face else 'off'}"
        elif key == ord("t"):
            cfg.use_tracking = not cfg.use_tracking
            track.bbox = None
            msg = f"tracking {'on' if cfg.use_tracking else 'off'}"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
