"""Manual ground truth: watch the live stream and press a key every time an object crosses the line.

    python -m tools.manual_counter CAMERA_ID [--out manual/cam2.csv]

Keys (lowercase = direction "in", UPPERCASE = "out"; the arrow on screen shows "in"):
    c car   t truck   b bus   m motorcycle   p person   y bicycle
    Backspace = undo last   q = quit
Only CSV rows (UTC time, camera, class, direction) are stored - never video or images.
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
from collections import Counter
from pathlib import Path

import cv2

from traffic.config import DEFAULT_CONFIG_PATH, load_config
from traffic.db import utc_now
from traffic.stream import Stream
from traffic.worker import draw_line

KEYS = {"c": "car", "t": "truck", "b": "bus", "m": "motorcycle", "p": "person", "y": "bicycle"}
FIELDS = ["ts_utc", "camera_id", "class_name", "direction"]


def key_to_event(ch: str) -> tuple[str, str] | None:
    """'c' -> ('car', 'in'), 'C' -> ('car', 'out'), anything else -> None."""
    if ch.lower() in KEYS:
        return KEYS[ch.lower()], "out" if ch.isupper() else "in"
    return None


def save(path: Path, rows: list[list[str]]) -> None:
    """Rewrite the whole file: simple, and makes undo trivial (files are small)."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerows(rows)


def draw_counts(frame, counts: Counter) -> None:
    y = 25
    for name in KEYS.values():
        text = f"{name:<10} in {counts[(name, 'in')]:>3}  out {counts[(name, 'out')]:>3}"
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 24


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("camera")
    ap.add_argument("--out", help="CSV file (default: manual/cam<ID>_<time>.csv)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = ap.parse_args(argv)
    cam = load_config(args.config).camera(args.camera)

    started = utc_now()
    out = Path(args.out or f"manual/cam{cam.id}_{started[:19].replace(':', '-')}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[str]] = []
    counts: Counter = Counter()
    stop = threading.Event()
    print(__doc__)

    for frame in Stream(cam.url, stop=stop).frames():
        draw_line(frame, cam.line)
        draw_counts(frame, counts)
        cv2.imshow(f"manual count - camera {cam.id} (q = quit)", frame)
        key = cv2.waitKey(30) & 0xFF  # ~ real-time playback for a 25 fps stream
        if key == ord("q"):
            stop.set()
        elif key in (8, 127) and rows:
            _, _, cls, direction = rows.pop()
            counts[(cls, direction)] -= 1
            save(out, rows)
        elif key != 255 and (event := key_to_event(chr(key))):
            rows.append([utc_now(), cam.id, *event])
            counts[event] += 1
            save(out, rows)
    cv2.destroyAllWindows()

    save(out, rows)
    print(f"Saved {len(rows)} crossings to {out}")
    print(f"Evaluate with:\n  python -m tools.evaluate {out} --start {started} --end {utc_now()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
