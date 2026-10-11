"""Phase 2: calibration UI. Drag a box on your hand to sample skin color,
tune with the trackbars, s=save, l=load, q=quit."""
import argparse

import cv2

from utils.utils_ import (MAIN_WIN, SampleState, create_trackbars,
                          detect_hand, draw_detection, draw_hud, load_config,
                          open_camera, read_frame, read_trackbars, sample_skin,
                          save_config, set_trackbars, setup_sampling, show)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    args = ap.parse_args()

    cap = open_camera(args.cam)
    cfg = load_config()
    state = SampleState()
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
        read_trackbars(cfg)

        det, mask = detect_hand(frame, cfg)
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
            msg = "loaded config.json"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
