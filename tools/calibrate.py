"""Choose the counting line: shows ONE live frame, you click two points, the YAML is printed.

    python -m tools.calibrate CAMERA_ID

Click the start and end of the line across the road, then press Enter to accept
(r = reset, q = quit). The frame stays in RAM only and is never saved.
"""

from __future__ import annotations

import argparse
import sys
import threading

import cv2
import numpy as np

from traffic.config import DEFAULT_CONFIG_PATH, load_config
from traffic.stream import Stream
from traffic.worker import draw_line


def grab_frame(url: str) -> np.ndarray:
    stop = threading.Event()
    for frame in Stream(url, stop=stop).frames():
        stop.set()
        return frame
    raise RuntimeError(f"no frame from {url}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("camera")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = ap.parse_args(argv)
    cam = load_config(args.config).camera(args.camera)

    print(f"Connecting to camera {cam.id} ...")
    frame = grab_frame(cam.url)
    points: list[tuple[int, int]] = []

    def on_click(event: int, x: int, y: int, *_: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    window = f"camera {cam.id}: click 2 points, Enter = accept, r = reset, q = quit"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)
    while True:
        view = frame.copy()
        for p in points:
            cv2.circle(view, p, 5, (0, 255, 0), -1)
        if len(points) == 2:
            draw_line(view, (points[0], points[1]))
        cv2.imshow(window, view)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            return 1
        if key == ord("r"):
            points.clear()
        if key in (10, 13) and len(points) == 2:
            break
    cv2.destroyAllWindows()

    height, width = frame.shape[:2]
    (x1, y1), (x2, y2) = points
    print(f"\nStream resolution: {width}x{height}")
    print(f"Put this under camera '{cam.id}' in config.yaml (the arrow showed the 'in' side):\n")
    print(f"    line: [[{x1}, {y1}], [{x2}, {y2}]]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
