"""Open a live stream and yield frames forever, reconnecting with exponential backoff.

The URL must be one that OpenCV/FFmpeg opens directly (HLS .m3u8, RTSP, a file...).
Frames are only yielded (kept in RAM); this module never writes anything to disk.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Iterator
from typing import Protocol

import cv2
import numpy as np

log = logging.getLogger(__name__)

MIN_BACKOFF_S = 2.0
MAX_BACKOFF_S = 60.0

# TCP for RTSP and a 10 s socket timeout, so a dead stream fails instead of hanging forever.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|timeout;10000000|rw_timeout;10000000")


class StopSignal(Protocol):
    """threading.Event or multiprocessing.Event."""

    def is_set(self) -> bool: ...
    def wait(self, timeout: float | None = None) -> bool: ...


def open_capture(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class Stream:
    """Iterate over `frames()` to get every `stride`-th frame of a live stream.

    `on_status` is called with "up", "down" or "reconnecting" whenever the state changes.
    `stop` ends the iteration (also interrupts the backoff wait).
    `session` grows by one on every successful (re)connection, so the consumer can tell
    that the frames after a reconnect do not continue the earlier ones.
    """

    def __init__(
        self,
        url: str,
        stride: int = 1,
        on_status: Callable[[str], None] = lambda status: None,
        stop: StopSignal | None = None,
        opener: Callable[[str], cv2.VideoCapture] = open_capture,
    ) -> None:
        self.url = url
        self.stride = stride
        self.on_status = on_status
        self.stop: StopSignal = stop or threading.Event()
        self.opener = opener
        self.status = ""
        self.session = 0

    def frames(self) -> Iterator[np.ndarray]:
        backoff = MIN_BACKOFF_S
        while not self.stop.is_set():
            cap = self._try_open()
            if cap is None:
                self._set_status("reconnecting")
                log.warning("cannot open %s, retrying in %.0fs", self.url, backoff)
                self.stop.wait(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF_S)
                continue
            backoff = MIN_BACKOFF_S
            self.session += 1
            self._set_status("up")
            try:
                yield from self._read_until_failure(cap)
            finally:
                cap.release()
            if not self.stop.is_set():
                self._set_status("down")

    def _try_open(self) -> cv2.VideoCapture | None:
        try:
            cap = self.opener(self.url)
        except Exception as err:  # network / FFmpeg errors must never kill the worker
            log.warning("resolving %s failed: %s", self.url, err)
            return None
        if cap.isOpened():
            return cap
        cap.release()
        return None

    def _read_until_failure(self, cap: cv2.VideoCapture) -> Iterator[np.ndarray]:
        while not self.stop.is_set():
            # grab() fetches a frame without converting it, so skipped frames are cheap.
            for _ in range(self.stride - 1):
                if not cap.grab():
                    return
            ok, frame = cap.read()
            if not ok or frame is None:
                return
            yield frame

    def _set_status(self, status: str) -> None:
        if status != self.status:
            self.status = status
            log.info("stream %s is %s", self.url, status)
            self.on_status(status)
