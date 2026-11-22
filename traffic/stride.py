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
"""

from __future__ import annotations

from dataclasses import dataclass

TARGET_LOAD = 0.8      # keep 20 % headroom for busier scenes and other processes
MAX_STRIDE = 10
START_STRIDE = 3       # safe on the server with 4 cameras; adapts from there
MIN_VIDEO_S = 5.0      # windows with less video than this say too little to act on


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
