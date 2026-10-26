import argparse
import time
import cv2
from utils.utils_ import (MAIN_WIN, BgState, Control, DetectorConfig, Drone, FaceState, HandFilter, SampleState, TrackState, create_trackbars, detect_hand, draw_detection, draw_faces, draw_game, draw_hud, draw_rois, draw_status, filter_hand, game_running, get_frame, hand_to_control, init_game, load_config, open_camera, read_trackbars, sample_skin, save_config, set_roi, set_trackbars, setup_sampling, show, start_bg_capture, start_grabber, stop_grabber, update_bg, update_drone, update_faces)

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
    hf = HandFilter()
    drone = Drone()
    ctrl = Control()
    msg = ""
    cv2.namedWindow(MAIN_WIN)
    create_trackbars(cfg)
    setup_sampling(state)
    screen, clock, font = init_game()
    grab = start_grabber(cap)
    last_id, last_t = 0, time.time()
    running = True

    while running:
        frame, last_id = get_frame(grab, last_id)
        if frame is not None:
            last_t = time.time()
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
            det = filter_hand(hf, det, cfg)
            ctrl = hand_to_control(det, cfg, frame.shape)
            draw_rois(frame, cfg, search)
            draw_faces(frame, face)
            draw_detection(frame, det)
            draw_status(frame, hf, cfg)
            draw_hud(frame, state, msg)

            key = show(frame, mask)
            if key == ord("q"):
                running = False
            elif key == ord("s"):
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
        elif time.time() - last_t > 0.5:
            ctrl = Control()  # camera stalled -> neutral

        running = game_running() and running
        update_drone(drone, ctrl, clock.tick(60) / 1000)
        draw_game(screen, font, drone, ctrl)

    stop_grabber(grab)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
