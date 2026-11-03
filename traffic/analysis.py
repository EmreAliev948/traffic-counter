"""Coverage-aware analysis of the counts (pure pandas; used by the dashboard and tools.report).

The key rule (theory, 1.5.1 and 1.5.3): *no data is not zero traffic*. Every rate is
computed only over the time in which the camera was really analysing frames, which is
reconstructed from the stream_health table:

- a worker writes an "up" row when the stream opens and then every ~30 s while frames flow;
- an "up" row therefore vouches for the time until the next row, but at most MAX_GAP;
- "down" / "reconnecting" / "stopped" rows, or missing heartbeats (process killed,
  laptop asleep), end the covered period.

Hours with less than MIN_COVERED_MIN operating minutes get NaN instead of a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import tzinfo

import numpy as np
import pandas as pd

from . import db

MAX_GAP = pd.Timedelta(seconds=90)  # 3 missed heartbeats (worker.HEALTH_EVERY_S = 30 s) -> no data
MIN_COVERED_MIN = 30.0              # an hour needs >= 30 operating minutes to get a rate
DAY_HOURS = (7, 19)                 # local [start, end) hours treated as "day"
MIN_HOURS_DECOMPOSE = 48            # need at least two full days for a daily profile
MIN_DAYS_WEEKLY = 14                # each weekday must be seen at least twice for a weekly profile
Z_THRESHOLD = 3.0

REASONS = {
    "down": "потокът прекъсна",
    "reconnecting": "повторно свързване",
    "stopped": "програмата е спряна",
    "up": "няма сигнал от процеса (не е работил)",
}
COUNT_COLUMNS = ["camera_id", "minute_utc", "class_name", "direction", "n"]


def to_utc(values) -> pd.Series | pd.DatetimeIndex:
    """Parse the database's ISO-8601 'Z' strings."""
    return pd.to_datetime(values, utc=True, format="ISO8601")


# --------------------------------------------------------------------------- coverage

def up_intervals(health: pd.DataFrame) -> pd.DataFrame:
    """[start, end) periods in which each camera was analysing frames (columns camera_id, start, end)."""
    rows: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    if health.empty:
        return pd.DataFrame(rows, columns=["camera_id", "start", "end"])
    h = health.assign(ts=to_utc(health["ts_utc"])).sort_values(["camera_id", "ts"], kind="stable")
    for cam, g in h.groupby("camera_id", sort=True):
        times, statuses = g["ts"].tolist(), g["status"].tolist()
        current: list[pd.Timestamp] | None = None
        for i, (t, status) in enumerate(zip(times, statuses, strict=True)):
            if status != "up":
                continue
            end = t + MAX_GAP if i + 1 == len(times) else min(times[i + 1], t + MAX_GAP)
            if current is not None and t <= current[1]:
                current[1] = max(current[1], end)
            else:
                if current is not None:
                    rows.append((cam, current[0], current[1]))
                current = [t, end]
        if current is not None:
            rows.append((cam, current[0], current[1]))
    return pd.DataFrame(rows, columns=["camera_id", "start", "end"])


def minute_coverage(
    intervals: pd.DataFrame, counts: pd.DataFrame, cameras: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Fraction (0..1) of every minute in [start, end) in which each camera was running.

    Index = minute (UTC), columns = camera ids. A minute that contains a crossing
    event is always covered: the event itself proves the camera was working.
    """
    minutes = pd.date_range(start.floor("min"), end, freq="min", inclusive="left")
    cov = np.zeros((len(minutes), len(cameras)))
    if len(minutes) == 0:
        return pd.DataFrame(cov, index=minutes, columns=cameras)
    col = {c: j for j, c in enumerate(cameras)}
    one_min = pd.Timedelta(minutes=1)
    for cam, s, e in intervals[intervals["camera_id"].isin(cameras)].itertuples(index=False):
        s, e = max(s, minutes[0]), min(e, end)
        if s >= e:
            continue
        idx = pd.date_range(s.floor("min"), e, freq="min", inclusive="left")
        overlap = (np.minimum(idx + one_min, e) - np.maximum(idx, s)) / one_min
        pos = ((idx - minutes[0]) // one_min).astype(int)
        np.add.at(cov[:, col[cam]], pos, np.asarray(overlap, dtype=float))
    if not counts.empty:
        t = pd.DatetimeIndex(to_utc(counts["minute_utc"])).floor("min")
        pos = np.asarray((t - minutes[0]) // one_min, dtype=int)
        cols = counts["camera_id"].map(col)
        ok = (pos >= 0) & (pos < len(minutes)) & cols.notna().to_numpy()
        cov[pos[ok], cols[ok].astype(int).to_numpy()] = 1.0
    return pd.DataFrame(np.clip(cov, 0.0, 1.0), index=minutes, columns=cameras)


def covered_minutes_per_hour(coverage: pd.DataFrame) -> pd.DataFrame:
    """Operating minutes (0..60) per hour (UTC index) and camera."""
    return coverage.resample("h").sum()


# --------------------------------------------------------------------------- rates

def hourly_counts(counts: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Events per UTC hour; columns = camera ids (by=None) or the values of column `by`."""
    if counts.empty:
        return pd.DataFrame()
    c = counts.assign(hour=to_utc(counts["minute_utc"]).dt.floor("h"))
    return c.pivot_table(index="hour", columns=by or "camera_id", values="n", aggfunc="sum", fill_value=0)


def hourly_rates(counts: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    """Crossings per operating hour for each camera (hours x cameras); NaN where coverage is too low."""
    cov_h = covered_minutes_per_hour(coverage)
    n = hourly_counts(counts).reindex(index=cov_h.index, columns=cov_h.columns, fill_value=0)
    rate = n / cov_h.where(cov_h > 0) * 60.0
    return rate.where(cov_h >= MIN_COVERED_MIN)


def mean_rate(rates: pd.DataFrame, tz: tzinfo) -> pd.Series:
    """Average crossings per hour per camera over the given cameras, in local time; NaN = no data."""
    return rates.mean(axis=1, skipna=True).tz_convert(tz)


def heatmap(series: pd.Series) -> pd.DataFrame:
    """Mean rate for weekday (0 = Monday) x hour of day; cells without data stay NaN."""
    s = series.dropna()
    grid = s.groupby([s.index.weekday, s.index.hour]).mean().unstack()
    return grid.reindex(index=range(7), columns=range(24))


# --------------------------------------------------------------------------- time-series methods

def decompose(series: pd.Series) -> pd.DataFrame | None:
    """Additive decomposition of an hourly series: observed = trend + daily + weekly + residual.

    trend    - centred 24-hour moving average (removes the daily cycle);
    daily    - mean of (observed - trend) for each hour of the day;
    weekly   - mean of what is left for each weekday (only with >= 14 days of data,
               otherwise every weekday would be seen once and absorb all the noise);
    residual - the rest; used for anomaly detection.
    Returns None when there are fewer than MIN_HOURS_DECOMPOSE hours with data.
    """
    if series.notna().sum() < MIN_HOURS_DECOMPOSE:
        return None
    s = series.asfreq("h")
    # Full 24-hour windows only (classical decomposition): a partial window at the edges or
    # next to a gap would mix in part of the daily cycle and bend the trend.
    trend = s.rolling(24, center=True, min_periods=24).mean()
    detrended = s - trend
    daily = detrended.groupby(s.index.hour).transform("mean")
    days_seen = s.dropna().index.normalize().nunique()
    weekly_used = days_seen >= MIN_DAYS_WEEKLY
    if weekly_used:
        weekly = (detrended - daily).groupby(s.index.weekday).transform("mean")
    else:
        weekly = pd.Series(0.0, index=s.index)
    out = pd.DataFrame({
        "observed": s, "trend": trend, "daily": daily.where(s.notna()),
        "weekly": weekly.where(s.notna()), "residual": s - trend - daily - weekly,
    })
    out.attrs["weekly_used"] = weekly_used
    return out


def anomalies(decomposition: pd.DataFrame | None, threshold: float = Z_THRESHOLD) -> pd.DataFrame:
    """Hours whose residual has |z-score| >= threshold (columns: observed, expected, z)."""
    columns = ["observed", "expected", "z"]
    if decomposition is None:
        return pd.DataFrame(columns=columns)
    r = decomposition["residual"].dropna()
    if len(r) < 24 or not r.std() > 0:
        return pd.DataFrame(columns=columns)
    z = (r - r.mean()) / r.std()
    hit = z[z.abs() >= threshold].index
    d = decomposition.loc[hit]
    return pd.DataFrame({
        "observed": d["observed"],
        "expected": d["trend"] + d["daily"] + d["weekly"],
        "z": z.loc[hit],
    })


def is_day(index: pd.DatetimeIndex, day_hours: tuple[int, int] = DAY_HOURS) -> np.ndarray:
    return (index.hour >= day_hours[0]) & (index.hour < day_hours[1])


def compare_periods(series: pd.Series, day_hours: tuple[int, int] = DAY_HOURS) -> pd.DataFrame:
    """Mean crossings per hour (per camera) for day / night and weekday / weekend."""
    s = series.dropna()
    a, b = day_hours
    groups = {
        f"Ден ({a:02d}–{b:02d} ч.)": is_day(s.index, day_hours),
        "Нощ": ~is_day(s.index, day_hours),
        "Делник (пн–пт)": s.index.weekday < 5,
        "Почивен ден (сб–нд)": s.index.weekday >= 5,
    }
    rows = [(name, s[mask].mean() if mask.any() else np.nan, int(mask.sum())) for name, mask in groups.items()]
    return pd.DataFrame(rows, columns=["период", "средно на час", "часове с данни"])


def class_shares(counts: pd.DataFrame, tz: tzinfo, day_hours: tuple[int, int] = DAY_HOURS) -> pd.DataFrame:
    """Share (%) of each class during the day and during the night (rows = class)."""
    if counts.empty:
        return pd.DataFrame()
    t = to_utc(counts["minute_utc"]).dt.tz_convert(tz)
    period = np.where(is_day(pd.DatetimeIndex(t), day_hours), "ден", "нощ")
    table = counts.assign(period=period).pivot_table(
        index="class_name", columns="period", values="n", aggfunc="sum", fill_value=0)
    return (table / table.sum() * 100).round(1)


# --------------------------------------------------------------------------- outages

def outages(
    health: pd.DataFrame, intervals: pd.DataFrame, until: pd.Timestamp, since: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Periods without data for each camera, with the reason taken from the last status row.

    Covers stream failures ("down", "reconnecting"), a clean stop ("stopped") and a
    crashed / killed process (heartbeats just stop). `end` is NaT if it lasts until `until`.
    Periods are clipped to start no earlier than `since`.
    """
    columns = ["camera_id", "start", "end", "minutes", "reason"]
    rows = []
    if health.empty:
        return pd.DataFrame(rows, columns=columns)
    h = health.assign(ts=to_utc(health["ts_utc"])).sort_values(["camera_id", "ts"], kind="stable")
    for cam, g in h.groupby("camera_id", sort=True):
        iv = intervals[intervals["camera_id"] == cam].sort_values("start")
        gaps, cursor = [], g["ts"].iloc[0]
        for s, e in zip(iv["start"], iv["end"], strict=True):
            if s > cursor:
                gaps.append((cursor, s))
            cursor = max(cursor, e)
        if cursor < until:
            gaps.append((cursor, pd.NaT))
        for s, e in gaps:
            if since is not None:
                if not pd.isna(e) and e <= since:
                    continue
                s = max(s, since)
            before = g[g["ts"] <= s]
            status = before["status"].iloc[-1] if len(before) else g["status"].iloc[0]
            minutes = round(((until if pd.isna(e) else e) - s).total_seconds() / 60, 1)
            rows.append((cam, s, e, minutes, REASONS.get(status, status)))
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------- one-stop preparation

@dataclass
class Window:
    """Everything the dashboard and the report need for one time window and set of cameras."""

    counts: pd.DataFrame     # COUNT_COLUMNS, only the chosen cameras
    health: pd.DataFrame
    intervals: pd.DataFrame
    coverage: pd.DataFrame   # minute x camera, 0..1
    rates: pd.DataFrame      # hour x camera, crossings per operating hour (NaN = no data)
    start: pd.Timestamp
    end: pd.Timestamp        # min(window end, now)

    @property
    def covered_hours(self) -> float:
        return float(self.coverage.to_numpy().sum() / 60.0)

    @property
    def coverage_pct(self) -> float:
        possible = self.coverage.size
        return 100.0 * self.coverage.to_numpy().sum() / possible if possible else 0.0


def prepare(
    counts: pd.DataFrame, health: pd.DataFrame, cameras: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> Window:
    now = pd.Timestamp.now(tz="UTC").ceil("min")
    end = min(end, now)
    counts = counts[counts["camera_id"].isin(cameras)].reset_index(drop=True)
    health = health[health["camera_id"].isin(cameras)].reset_index(drop=True)
    intervals = up_intervals(health)
    coverage = minute_coverage(intervals, counts, cameras, start, end)
    return Window(counts, health, intervals, coverage, hourly_rates(counts, coverage), start, end)


def load_window(conn, cameras: list[str] | None, start: pd.Timestamp, end: pd.Timestamp) -> Window:
    """Read counts and stream health for [start, end) from SQLite and prepare them."""
    start_utc, end_utc = db.format_utc(start.to_pydatetime()), db.format_utc(end.to_pydatetime())
    counts = pd.DataFrame(db.minute_counts(conn, start_utc, end_utc), columns=COUNT_COLUMNS)
    health = pd.DataFrame(db.health_rows(conn, start_utc, end_utc), columns=["camera_id", "ts_utc", "status", "fps"])
    if cameras is None:  # every camera that is registered or has data in the window
        cameras = sorted(set(db.camera_names(conn)) | set(counts["camera_id"]) | set(health["camera_id"]))
    return prepare(counts, health, cameras, start, end)
