"""One process per camera: read frame -> YOLO + ByteTrack -> line counter -> buffered SQLite writes.

The main process (`supervise`) starts one worker per enabled camera, restarts a worker
that died, and stops all of them cleanly on Ctrl+C.

Privacy: a frame exists only as a NumPy array inside one loop iteration. Only metadata
(time, camera, track id, class, direction, confidence) is written to the database.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from . import db
from .config import Camera, Config, Defaults
from .counter import LineCounter, TrackedBox
from .model import prepare_model
from .stream import StopSignal, Stream
from .stride import START_STRIDE, StrideHistory, choose_stride, measured_load

log = logging.getLogger(__name__)

FLUSH_EVERY_S = 2.0
FLUSH_EVERY_N = 100
HEALTH_EVERY_S = 30.0
RESTART_DELAY_S = 10.0

TrackFn = Callable[[np.ndarray], list[TrackedBox]]
WindowFn = Callable[[float, int, float], str]  # (wall_s, analysed, analyse_s) -> text for the log


class StrideControl:
    """Called once per health window: reads the stream's stats and, in auto mode, sets its stride."""

    def __init__(self, stream: Stream, auto: bool) -> None:
        self.stream = stream
        self.auto = auto
        self.session = stream.session
        self.history = StrideHistory()

    def __call__(self, wall_s: float, analysed: int, analyse_s: float) -> str:
        stats = self.stream.take_stats()
        reconnected, self.session = self.stream.session != self.session, self.stream.session
        if stats.video_s <= 0:
            return f"stride {self.stream.stride}"
        load = measured_load(stats, wall_s)
        if self.auto and not reconnected:  # a reconnect's waiting would distort the load
            proposed = choose_stride(self.stream.stride, stats, wall_s, analysed, analyse_s)
            self.stream.stride = self.history.next(self.stream.stride, proposed)
        return f"stride {self.stream.stride}, load {load:.2f}"


class EventBuffer:
    """Collects events and writes them in one transaction every 2 s or 100 events."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.events: list[db.CrossingEvent] = []
        self.last_flush = time.monotonic()

    def add(self, event: db.CrossingEvent) -> None:
        self.events.append(event)

    def flush_if_due(self) -> None:
        if len(self.events) >= FLUSH_EVERY_N or time.monotonic() - self.last_flush >= FLUSH_EVERY_S:
            self.flush()

    def flush(self) -> None:
        if self.events:
            db.insert_events(self.conn, self.events)
            self.events = []
        self.last_flush = time.monotonic()


def load_model(defaults: Defaults, model_path: str | None = None):
    """Load the prepared model (an export, see traffic/model.py), else the PyTorch weights,
    else the fallback (e.g. old ultralytics)."""
    from ultralytics import YOLO

    names = dict.fromkeys(n for n in (model_path, defaults.model, defaults.fallback_model) if n)
    for name in names:
        try:
            return YOLO(name, task="detect")
        except Exception as err:
            log.warning("cannot load model %s: %s", name, err)
    raise RuntimeError(f"neither {defaults.model} nor {defaults.fallback_model} could be loaded")


class YoloTracker:
    """YOLO + ByteTrack as a callable: frame -> list of tracked boxes.

    ByteTrack matches in two stages. Stage 1 matches existing tracks to high-score
    boxes (>= 0.25) using Kalman-predicted positions and IoU. Stage 2 takes the tracks
    still unmatched and tries the LOW-score boxes (0.1-0.25). A car half hidden behind
    a bus often scores only ~0.15; stage 2 keeps its track alive instead of losing it
    and starting a new id later, which would cause double counts or missed crossings.
    That is why the detector's `conf` is set low (0.1): with 0.25 the low-score boxes
    would be thrown away before ByteTrack ever sees them. New tracks still need >= 0.25,
    so weak boxes can only extend existing tracks, never create false ones.
    """

    def __init__(self, model, defaults: Defaults) -> None:
        self.model = model
        self.defaults = defaults
        self.names: dict[int, str] = model.names

    def __call__(self, frame: np.ndarray) -> list[TrackedBox]:
        d = self.defaults
        boxes: list[TrackedBox] = []
        # persist=True keeps the tracker state between calls; stream=True returns a
        # generator so results are consumed immediately and never accumulate.
        for result in self.model.track(
            frame, stream=True, persist=True, tracker="bytetrack.yaml", conf=d.conf,
            imgsz=d.imgsz, classes=list(d.classes), verbose=False, save=False,
        ):
            b = result.boxes
            if b is None or b.id is None:  # no confirmed tracks in this frame
                continue
            rows = zip(b.id.int().tolist(), b.cls.int().tolist(), b.conf.tolist(), b.xyxy.tolist(), strict=True)
            for tid, cls, conf, xyxy in rows:
                boxes.append(TrackedBox(tid, self.names[cls], float(conf), tuple(xyxy)))
        return boxes

    def reset(self) -> None:
        """Forget all tracks (after a reconnect the old positions are meaningless).

        Track ids restart at 1, which is safe: the line counter is reset at the same
        time and event ids also contain the stream session number.
        """
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()


def make_tracker(model, defaults: Defaults) -> YoloTracker:
    return YoloTracker(model, defaults)


def process_frames(
    frames: Iterable[np.ndarray],
    track: TrackFn,
    cam: Camera,
    conn: sqlite3.Connection,
    run_id: str,
    preview: bool = False,
    session: Callable[[], int] | None = None,
    margin_px: float = 6.0,
    on_window: WindowFn | None = None,
) -> None:
    """The main loop. `frames` is a Stream in production and a plain list in tests.

    `session()` returns the stream's connection number. When it changes, the stream was
    reconnected: the tracker and the line counter are reset, otherwise an old track
    could be matched to a new object somewhere else and produce a false crossing.

    `on_window` is called every HEALTH_EVERY_S with the window's wall time, the number of
    analysed frames and the time spent analysing them (model + counting).
    """
    counter = LineCounter(cam.line, margin_px=margin_px)
    buffer = EventBuffer(conn)
    analysed, analyse_s, window_start = 0, 0.0, time.monotonic()
    current_session = session() if session else 0
    try:
        for frame in frames:
            if session is not None and session() != current_session:
                reconnected = current_session != 0  # 0 -> 1 is just the first connection
                current_session = session()
                if reconnected:
                    counter.reset()
                    reset = getattr(track, "reset", None)
                    if reset is not None:
                        reset()
                    log.info("camera %s: stream reconnected (session %d), tracks reset", cam.id, current_session)
            started = time.monotonic()
            boxes = track(frame)
            crossings = counter.update(boxes)
            analyse_s += time.monotonic() - started
            for c in crossings:
                buffer.add(db.CrossingEvent(
                    event_id=db.make_event_id(cam.id, f"{run_id}-{current_session}", c.track_id, c.direction),
                    camera_id=cam.id, ts_utc=db.utc_now(), track_id=c.track_id,
                    class_name=c.class_name, direction=c.direction, confidence=round(c.confidence, 3),
                ))
                log.info("camera %s: %s #%d %s", cam.id, c.class_name, c.track_id, c.direction)
            buffer.flush_if_due()
            if preview and not show_preview(frame, boxes, cam):
                break
            del frame  # drop the only reference: the image is freed right away

            analysed += 1
            elapsed = time.monotonic() - window_start
            if elapsed >= HEALTH_EVERY_S:
                fps = analysed / elapsed
                db.insert_health(conn, cam.id, "up", round(fps, 2))
                extra = f", {on_window(elapsed, analysed, analyse_s)}" if on_window else ""
                log.info("camera %s: %.1f analysed frames/s%s, %d active tracks", cam.id, fps, extra, counter.active_tracks)
                analysed, analyse_s, window_start = 0, 0.0, time.monotonic()
    finally:
        buffer.flush()


def show_preview(frame: np.ndarray, boxes: list[TrackedBox], cam: Camera) -> bool:
    """Draw boxes and the line in a window (on screen only). Returns False when 'q' is pressed."""
    import cv2

    draw_line(frame, cam.line)
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{b.class_name} #{b.track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow(f"camera {cam.id} (q = quit)", frame)
    return cv2.waitKey(1) & 0xFF != ord("q")


def draw_line(frame: np.ndarray, line: tuple[tuple[int, int], tuple[int, int]]) -> None:
    """Draw the counting line plus an arrow pointing to the "in" side (display only)."""
    import cv2

    (ax, ay), (bx, by) = line
    mx, my = (ax + bx) // 2, (ay + by) // 2
    length = max(1.0, ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5)
    # (-(by-ay), bx-ax) is perpendicular to the line and points to its right ("in") side.
    tip = (int(mx - 40 * (by - ay) / length), int(my + 40 * (bx - ax) / length))
    cv2.line(frame, (ax, ay), (bx, by), (0, 255, 255), 2)
    cv2.arrowedLine(frame, (mx, my), tip, (0, 255, 255), 2, tipLength=0.3)
    cv2.putText(frame, "in", tip, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def run_camera(
    cam: Camera, defaults: Defaults, stop: StopSignal, threads: int, preview: bool = False,
    model_path: str | None = None,
) -> None:
    """Entry point of a worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is handled by the main process via `stop`
    signal.signal(signal.SIGTERM, _exit_on_sigterm)  # systemd may signal every process, not only the main one
    setup_logging()
    import torch

    torch.set_num_threads(threads)  # share CPU cores fairly between the camera processes
    run_id = uuid.uuid4().hex  # new per process start, so restarted workers never reuse event ids
    conn = db.connect(defaults.db_path)

    def on_status(status: str) -> None:
        db.insert_health(conn, cam.id, status)

    stream = Stream(cam.url, defaults.vid_stride or START_STRIDE, on_status, stop)
    control = StrideControl(stream, auto=defaults.vid_stride is None)
    track = make_tracker(load_model(defaults, model_path), defaults)
    log.info("camera %s started (run %s, %d threads, model %s)", cam.id, run_id[:8], threads, model_path or defaults.model)
    try:
        process_frames(
            stream.frames(), track, cam, conn, run_id, preview,
            session=lambda: stream.session, margin_px=defaults.line_margin_px, on_window=control,
        )
        stop.set()  # a normal return means 'q' in the preview window: end the whole run
    finally:
        try:
            # Marks the end of the covered period, so the report shows "stopped" instead
            # of silently treating the time after it as zero traffic.
            db.insert_health(conn, cam.id, "stopped")
        except sqlite3.Error as err:
            log.warning("camera %s: cannot write 'stopped' status: %s", cam.id, err)
        conn.close()
        log.info("camera %s stopped", cam.id)


def supervise(config: Config, only: list[str] | None = None, preview: bool = False) -> None:
    """Start one process per camera, restart dead ones, stop all on Ctrl+C / SIGTERM."""
    cameras = [config.camera(c) for c in only] if only else [c for c in config.cameras if c.enabled]
    if not cameras:
        raise SystemExit("no cameras to run (enable one in config.yaml or use --only)")
    conn = db.connect(config.defaults.db_path)
    db.init_db(conn)
    for cam in cameras:
        db.upsert_camera(conn, cam)
    conn.close()
    model_path = prepare_model(config.defaults)  # export once here, not in 4 processes at the same time

    ctx = mp.get_context("spawn")  # same behaviour on macOS, Linux and Windows
    stop = ctx.Event()
    threads = max(1, (os.cpu_count() or 2) // len(cameras))
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _raise_keyboard_interrupt)

    def start(cam: Camera) -> mp.process.BaseProcess:
        args = (cam, config.defaults, stop, threads, preview, model_path)
        p = ctx.Process(target=run_camera, args=args, name=f"cam-{cam.id}")
        p.start()
        return p

    procs = {cam.id: (cam, start(cam), time.monotonic()) for cam in cameras}
    log.info("running %d camera(s): %s; database %s", len(cameras), ", ".join(procs), Path(config.defaults.db_path))
    try:
        while not stop.wait(2.0):
            for cam_id, (cam, proc, started) in list(procs.items()):
                if not proc.is_alive() and time.monotonic() - started > RESTART_DELAY_S:
                    log.error("camera %s worker died (exit code %s), restarting", cam_id, proc.exitcode)
                    procs[cam_id] = (cam, start(cam), time.monotonic())
    except KeyboardInterrupt:
        log.info("stopping...")
    finally:
        stop.set()
        for _, proc, _ in procs.values():
            proc.join(timeout=15)
            if proc.is_alive():
                proc.kill()  # SIGTERM would only ask again (see _exit_on_sigterm); this one is stuck


def _exit_on_sigterm(*_: object) -> None:
    """SIGTERM in a camera process -> normal exit through the `finally` blocks, so the
    buffered events are written and a 'stopped' row marks the end of the covered period.

    Raising instead of calling stop.set() avoids the deadlock described below. Further
    SIGTERMs are ignored so they cannot interrupt that cleanup.
    """
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise SystemExit(0)


def _raise_keyboard_interrupt(*_: object) -> None:
    # Do NOT call stop.set() here: the handler may run inside stop.wait(), which holds
    # the same non-reentrant lock -> deadlock. Raising takes the normal Ctrl+C path.
    raise KeyboardInterrupt


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(processName)s %(levelname)s %(message)s")
