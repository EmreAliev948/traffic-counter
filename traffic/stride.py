"""Automatic frame stride: analyse as many frames as the CPU allows without falling behind.

Why not "always take the newest frame"? These HLS cameras deliver video in bursts: a whole
~7.5 s segment arrives at once, then nothing until the next one (measured 2026-11-23).
Taking only the newest frame would analyse a few frames per burst and skip seconds of video.
Instead every `stride`-th frame is analysed, in video order, and the stride is adapted.

Load = wall time the camera was busy / video time it processed. Below 1 it keeps up with the
live stream (the rest of the time it waits for the next segment); above 1 it falls behind.

    load(stride) ~= fps * seconds_per_analysed_frame / stride + other

`other` (decoding, reading, database) is whatever the measured load does not explain, so the
model always matches the measurement at the current stride. The new stride is the smallest
one whose predicted load fits under TARGET_LOAD. It may go up at once, but only down one step
per window, so a single quiet window cannot swing it.

The model assumes the time per analysed frame stays the same, but the cameras share the CPU:
when one analyses more frames, every camera's frames get slower. So a lower stride can turn
out heavier than predicted. `StrideHistory` remembers such a stride and does not retry it for
RETRY_AFTER_WINDOWS, which stops the camera from flipping between two strides every minute.
Each further failure of the same stride doubles that wait (up to MAX_RETRY_WINDOWS): on the
server stride 1 never held with 4 cameras, yet was retried every 10 min, and when several
cameras retried at once they slowed each other down enough to push stride 2 out as well.
Stepping down from a stride shows it held, so its failure count starts over.
"""

from __future__ import annotations

from dataclasses import dataclass

TARGET_LOAD = 0.8      # keep 20 % headroom for busier scenes and other processes
MAX_STRIDE = 10
START_STRIDE = 3       # safe on the server with 4 cameras; adapts from there
MIN_VIDEO_S = 5.0      # windows with less video than this say too little to act on
RETRY_AFTER_WINDOWS = 20  # a stride that proved too heavy is not tried again for 20 windows (10 min) ...
MAX_RETRY_WINDOWS = 240   # ... doubled after every further failure, up to 240 windows (2 h)


@dataclass(frozen=True)
class WindowStats:
    """What the stream saw during one measurement window."""

    frames: int     # frames decoded (analysed + skipped)
    video_s: float  # video time covered, from the frames' timestamps
    wait_s: float   # time spent waiting for data from the network


def measured_load(stats: WindowStats, wall_s: float) -> float:
    """Busy wall time per second of video: < 1 keeps up with the live stream, > 1 falls behind."""
    return max(0.0, wall_s - stats.wait_s) / stats.video_s if stats.video_s > 0 else 0.0


def choose_stride(current: int, stats: WindowStats, wall_s: float, analysed: int, analyse_s: float) -> int:
    """New stride after a window of `wall_s` seconds in which `analysed` frames took `analyse_s`."""
    if stats.video_s < MIN_VIDEO_S or analysed == 0 or stats.frames < analysed:
        return current
    if stats.video_s > 10 * wall_s:  # timestamps jumped (stream discontinuity): ignore this window
        return current
    fps = stats.frames / stats.video_s
    per_frame = analyse_s / analysed
    other = max(0.0, measured_load(stats, wall_s) - fps * per_frame / current)
    best = next((k for k in range(1, MAX_STRIDE + 1) if fps * per_frame / k + other <= TARGET_LOAD), MAX_STRIDE)
    return max(best, current - 1)  # up at once, down one step per window


class StrideHistory:
    """Remembers strides that proved too heavy, so the controller does not flip back to them."""

    def __init__(self) -> None:
        self.window = 0
        self._blocked_until: dict[int, int] = {}
        self._failures: dict[int, int] = {}

    def next(self, current: int, proposed: int) -> int:
        self.window += 1
        if proposed > current:  # `current` was too heavy: leave it alone for a while
            failures = self._failures.get(current, 0)
            self._failures[current] = failures + 1
            wait = min(RETRY_AFTER_WINDOWS * 2**failures, MAX_RETRY_WINDOWS)
            self._blocked_until[current] = self.window + wait
        elif proposed < current:
            if self._blocked_until.get(proposed, 0) > self.window:
                return current
            self._failures.pop(current, None)  # `current` held: it starts over
        return proposed
