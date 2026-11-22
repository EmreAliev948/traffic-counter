"""Pure line-crossing logic: tracked boxes in, crossing events out.

No OpenCV, no model, no database - so it is fully unit-testable with plain numbers.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

Point = tuple[float, float]


@dataclass(frozen=True)
class TrackedBox:
    """One tracked object in one frame (box in pixels: x1, y1, x2, y2)."""

    track_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]

    @property
    def anchor(self) -> Point:
        """Bottom-centre of the box: the point where the object touches the road."""
        x1, _, x2, y2 = self.box
        return (x1 + x2) / 2.0, y2


@dataclass(frozen=True)
class Crossing:
    track_id: int
    class_name: str
    direction: str  # "in" or "out"
    confidence: float


def side(a: Point, b: Point, p: Point) -> float:
    """Cross product of (b - a) x (p - a): > 0 right of a->b in image coords, < 0 left, 0 on the line."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """True if segment p1-p2 and segment q1-q2 touch or cross."""
    d1, d2 = side(q1, q2, p1), side(q1, q2, p2)
    d3, d4 = side(p1, p2, q1), side(p1, p2, q2)
    return d1 * d2 <= 0 and d3 * d4 <= 0


RIDEABLE = frozenset({"bicycle", "motorcycle"})
RIDER_OVERLAP = 0.3  # share of a person's box inside a bicycle/motorcycle box that makes them a rider


def overlap_share(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> float:
    """Fraction of `inner`'s area that lies inside `outer` (0..1)."""
    w = min(inner[2], outer[2]) - max(inner[0], outer[0])
    h = min(inner[3], outer[3]) - max(inner[1], outer[1])
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return 0.0 if w <= 0 or h <= 0 or area <= 0 else (w * h) / area


@dataclass
class _TrackState:
    ref_point: Point      # last anchor that was clearly (>= margin) on one side, or the first anchor
    ref_side: float       # sign of ref_point's side: -1, 0 (first seen inside the margin) or +1
    last_seen: int        # frame number
    counted: bool = False
    frames: int = 0
    rider_frames: int = 0
    class_score: dict[str, float] = field(default_factory=dict)  # sum of confidences per class
    class_hits: dict[str, int] = field(default_factory=dict)

    def observe(self, box: TrackedBox, riding: bool) -> None:
        self.frames += 1
        self.rider_frames += riding
        self.class_score[box.class_name] = self.class_score.get(box.class_name, 0.0) + box.confidence
        self.class_hits[box.class_name] = self.class_hits.get(box.class_name, 0) + 1

    def voted_class(self) -> tuple[str, float]:
        """The class with the highest summed confidence, and its mean confidence.

        YOLO often flickers between car / truck / bus on the same vehicle, so one frame's
        label is unreliable; the whole track's history is not.
        """
        cls = max(self.class_score, key=self.class_score.__getitem__)
        return cls, self.class_score[cls] / self.class_hits[cls]


class LineCounter:
    """Counts tracks whose bottom-centre crosses the segment line[0] -> line[1].

    "in"  = from the left side to the right side of the line (looking from line[0] to line[1]).
    "out" = the opposite.

    Rules that keep the count close to reality:
    * Each track is counted at most ONCE. A box that wobbles on the line (a queue at a
      checkpoint) would otherwise produce an "in" and an "out" for one vehicle.
    * The anchor must get at least `margin_px` pixels past the line before the crossing
      counts, so detection jitter of a few pixels never counts as a crossing.
    * The class is voted over the whole track (see `_TrackState.voted_class`).
    * A person riding a bicycle/motorcycle in most frames is not counted as a pedestrian:
      the bicycle/motorcycle track already counts them.

    Tracks not seen for `forget_after` frames are dropped, so memory stays bounded.
    """

    def __init__(self, line: tuple[Point, Point], forget_after: int = 90, margin_px: float = 6.0) -> None:
        self.a, self.b = line
        self.forget_after = forget_after
        self.margin_px = margin_px
        self._length = max(1e-9, ((self.b[0] - self.a[0]) ** 2 + (self.b[1] - self.a[1]) ** 2) ** 0.5)
        self.frame_no = 0
        self._tracks: dict[int, _TrackState] = {}

    def update(self, boxes: Iterable[TrackedBox]) -> list[Crossing]:
        self.frame_no += 1
        boxes = list(boxes)
        rideable = [b.box for b in boxes if b.class_name in RIDEABLE]
        crossings = []
        for box in boxes:
            riding = box.class_name == "person" and any(overlap_share(box.box, r) >= RIDER_OVERLAP for r in rideable)
            if (c := self._update_one(box, riding)) is not None:
                crossings.append(c)
        self._forget_old_tracks()
        return crossings

    def reset(self) -> None:
        self._tracks.clear()

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    def _signed_distance(self, p: Point) -> float:
        """Distance of p from the line in pixels: > 0 on the "in" (right) side, < 0 on the left."""
        return side(self.a, self.b, p) / self._length

    def _side_of(self, p: Point) -> float:
        """-1 / +1 when p is clearly on one side, 0 when it is within the margin."""
        d = self._signed_distance(p)
        return 0.0 if abs(d) < self.margin_px else (1.0 if d > 0 else -1.0)

    def _update_one(self, box: TrackedBox, riding: bool) -> Crossing | None:
        now = box.anchor
        s_now = self._side_of(now)
        state = self._tracks.get(box.track_id)
        if state is None:
            state = _TrackState(ref_point=now, ref_side=s_now, last_seen=self.frame_no)
            state.observe(box, riding)
            self._tracks[box.track_id] = state
            return None
        state.last_seen = self.frame_no
        state.observe(box, riding)
        if state.counted or s_now == 0:  # already counted, or still too close to the line to decide
            return None
        prev, prev_side = state.ref_point, state.ref_side
        state.ref_point, state.ref_side = now, s_now
        if prev_side == s_now:
            return None
        if prev_side == 0:
            # First seen inside the margin: it counts only if it was on the other side of the line.
            raw_prev = self._signed_distance(prev)
            if raw_prev == 0 or (raw_prev > 0) == (s_now > 0):
                return None
        # The side changed, but the object may have passed beside the segment's end.
        if not segments_intersect(prev, now, self.a, self.b):
            return None
        state.counted = True
        if state.rider_frames * 2 >= state.frames:
            return None  # a rider: the bicycle / motorcycle carrying them is counted instead
        direction = "in" if s_now > 0 else "out"
        class_name, confidence = state.voted_class()
        return Crossing(box.track_id, class_name, direction, confidence)

    def _forget_old_tracks(self) -> None:
        cutoff = self.frame_no - self.forget_after
        for track_id in [t for t, s in self._tracks.items() if s.last_seen < cutoff]:
            del self._tracks[track_id]
