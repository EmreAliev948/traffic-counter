"""Accuracy check: manual counts (CSV from tools.manual_counter) vs. automatic counts (SQLite).

    python -m tools.evaluate manual/cam2.csv [--start 2026-10-15T08:00:00Z --end 2026-10-15T08:15:00Z]
    python -m tools.evaluate manual/*.csv --summary reports/accuracy.csv    # several counts at once

Without --start/--end the window is the first..last manual click, widened by --lag
seconds on both sides (the automatic count is a few seconds behind a human).
Prints manual, auto, absolute error and % error per class and direction, plus the MAE.
Target: <= 10 % total error for cars in daylight.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

from traffic import db
from traffic.config import DEFAULT_CONFIG_PATH, load_config

Key = tuple[str, str]  # (class_name, direction)
TARGET_CAR_ERROR_PCT = 10.0  # success criterion: total car count within 10 % of the manual count


def load_manual(path: Path) -> tuple[str, Counter, list[str]]:
    """Returns (camera_id, counts per (class, direction), sorted timestamps)."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} has no counts")
    cameras = {r["camera_id"] for r in rows}
    if len(cameras) != 1:
        raise SystemExit(f"{path} mixes cameras {sorted(cameras)}; use one file per camera")
    counts = Counter((r["class_name"], r["direction"]) for r in rows)
    return cameras.pop(), counts, sorted(r["ts_utc"] for r in rows)


def shift(ts: str, seconds: float) -> str:
    return db.format_utc(datetime.fromisoformat(ts.replace("Z", "+00:00")) + timedelta(seconds=seconds))


def compare(manual: Counter, auto: dict[Key, int]) -> list[tuple[str, str, int, int, int, float | None]]:
    """One row per (class, direction): manual, auto, |error|, % error (None if manual == 0)."""
    rows = []
    for cls, direction in sorted(set(manual) | set(auto)):
        m, a = manual.get((cls, direction), 0), auto.get((cls, direction), 0)
        rows.append((cls, direction, m, a, abs(a - m), 100.0 * abs(a - m) / m if m else None))
    return rows


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}%"


def evaluate_file(path: Path, conn, start: str | None, end: str | None, lag: float) -> dict:
    """Print the per-class table for one manual CSV and return its summary row."""
    camera_id, manual, times = load_manual(path)
    w_start = shift(start or times[0], 0 if start else -lag)
    w_end = shift(end or times[-1], 0 if end else lag)
    auto = db.count_by_class_direction(conn, camera_id, w_start, w_end)

    rows = compare(manual, auto)
    print(f"{path}: camera {camera_id}, window {w_start} .. {w_end} (UTC)\n")
    print(f"{'class':<12}{'dir':<5}{'manual':>8}{'auto':>8}{'abs err':>9}{'% err':>9}")
    for cls, direction, m, a, err, rel in rows:
        print(f"{cls:<12}{direction:<5}{m:>8}{a:>8}{err:>9}{pct(rel):>9}")
    mae = sum(r[4] for r in rows) / len(rows)
    print(f"\nMAE over {len(rows)} class/direction cells: {mae:.2f}")

    m_total, a_total = sum(manual.values()), sum(auto.values())
    cars_manual = sum(n for (cls, _), n in manual.items() if cls == "car")
    cars_auto = sum(n for (cls, _), n in auto.items() if cls == "car")
    car_err = 100.0 * abs(cars_auto - cars_manual) / cars_manual if cars_manual else None
    if car_err is not None:
        verdict = "OK" if car_err <= TARGET_CAR_ERROR_PCT else "above target"
        print(f"Cars total: manual {cars_manual}, auto {cars_auto}, error {car_err:.1f}% "
              f"(target <= {TARGET_CAR_ERROR_PCT:g}%: {verdict})")
    print()
    return {
        "file": path.name, "camera": camera_id, "start": w_start, "end": w_end,
        "manual": m_total, "auto": a_total,
        "total_err_pct": 100.0 * abs(a_total - m_total) / m_total if m_total else None,
        "car_err_pct": car_err, "mae": mae,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manual", type=Path, nargs="+", help="CSV file(s) written by tools.manual_counter")
    ap.add_argument("--start", help="window start, UTC ISO-8601 (default: first click - lag)")
    ap.add_argument("--end", help="window end, UTC ISO-8601 (default: last click + lag)")
    ap.add_argument("--lag", type=float, default=5.0, help="seconds added around the clicks (default 5)")
    ap.add_argument("--summary", type=Path, help="also write the summary table to this CSV file")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = ap.parse_args(argv)

    with closing(db.connect(load_config(args.config).defaults.db_path, read_only=True)) as conn:
        results = [evaluate_file(p, conn, args.start, args.end, args.lag) for p in args.manual]

    if len(results) > 1 or args.summary:
        print("Summary (one row per manual count):")
        print(f"{'file':<34}{'cam':<5}{'manual':>7}{'auto':>7}{'total err':>11}{'car err':>9}{'MAE':>7}")
        for r in results:
            print(f"{r['file']:<34}{r['camera']:<5}{r['manual']:>7}{r['auto']:>7}"
                  f"{pct(r['total_err_pct']):>11}{pct(r['car_err_pct']):>9}{r['mae']:>7.2f}")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSummary written to {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
