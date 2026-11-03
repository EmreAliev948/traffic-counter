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


@dataclass
class _TrackState:
    point: Point          # last anchor that was NOT exactly on the line
    last_seen: int        # frame number
    counted: set[str] = field(default_factory=set)


class LineCounter:
    """Counts tracks whose bottom-centre crosses the segment line[0] -> line[1].

    "in"  = from the left side to the right side of the line (looking from line[0] to line[1]).
    "out" = the opposite. Each track is counted at most once per direction.
    Tracks not seen for `forget_after` frames are dropped, so memory stays bounded.
    """

    def __init__(self, line: tuple[Point, Point], forget_after: int = 90) -> None:
        self.a, self.b = line
        self.forget_after = forget_after
        self.frame_no = 0
        self._tracks: dict[int, _TrackState] = {}

    def update(self, boxes: Iterable[TrackedBox]) -> list[Crossing]:
        self.frame_no += 1
        crossings = [c for box in boxes if (c := self._update_one(box)) is not None]
        self._forget_old_tracks()
        return crossings

    def reset(self) -> None:
        self._tracks.clear()

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    def _update_one(self, box: TrackedBox) -> Crossing | None:
        now = box.anchor
        state = self._tracks.get(box.track_id)
        if state is None:
            self._tracks[box.track_id] = _TrackState(point=now, last_seen=self.frame_no)
            return None
        state.last_seen = self.frame_no
        s_now = side(self.a, self.b, now)
        if s_now == 0:  # exactly on the line: wait until it is clearly on one side
            return None
        s_prev = side(self.a, self.b, state.point)
        prev = state.point
        state.point = now
        if s_prev * s_now >= 0:  # same side, or the track was first seen exactly on the line
            return None
        # The sign changed, but the object may have passed beside the segment's end.
        if not segments_intersect(prev, now, self.a, self.b):
            return None
        direction = "in" if s_prev < 0 else "out"
        if direction in state.counted:
            return None
        state.counted.add(direction)
        return Crossing(box.track_id, box.class_name, direction, box.confidence)

    def _forget_old_tracks(self) -> None:
        cutoff = self.frame_no - self.forget_after
        for track_id in [t for t, s in self._tracks.items() if s.last_seen < cutoff]:
            del self._tracks[track_id]
