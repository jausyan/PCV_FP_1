import argparse
import time

import cv2

from utils.utils_ import (
    MAIN_WIN, Control, Drone, Session,
    apply_mouse_requests, create_trackbars, draw_game, draw_overlays, game_running,
    get_frame, handle_key, hand_to_control, init_game, load_config, open_camera,
    process_frame, setup_mouse, show, start_grabber, stop_grabber, update_drone,
)

STALL_SECONDS = 0.5  # no new camera frame for this long -> drone goes neutral


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    args = ap.parse_args()

    cap = open_camera(args.cam)
    session = Session(cfg=load_config())
    drone, ctrl = Drone(), Control()

    cv2.namedWindow(MAIN_WIN)
    create_trackbars(session.cfg)
    setup_mouse(session.mouse)
    screen, clock, font = init_game()
    grabber = start_grabber(cap)

    last_id, last_frame_time = 0, time.time()
    running = True
    while running:
        frame, last_id = get_frame(grabber, last_id)
        if frame is not None:
            last_frame_time = time.time()
            apply_mouse_requests(frame, session)
            hand, mask = process_frame(frame, session)
            ctrl = hand_to_control(hand, session.cfg, frame.shape)
            draw_overlays(frame, session, hand)
            running = handle_key(show(frame, mask), session)
        elif time.time() - last_frame_time > STALL_SECONDS:
            ctrl = Control()

        running = game_running() and running
        update_drone(drone, ctrl, clock.tick(60) / 1000)
        draw_game(screen, font, drone, ctrl)

    stop_grabber(grabber)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
