"""Export charts (PNG, matplotlib) and tables (CSV + summary.txt) for the written report.

    python -m tools.report [--from 2026-10-20] [--to 2026-10-26] [--camera 2 --camera 5] [--out reports]

Dates are local days (config timezone), both inclusive; default = the last 7 days.
Only aggregated counts are drawn - no video frames are involved. Uses the same
coverage-aware analysis as the dashboard (traffic/analysis.py).
"""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")  # no window: files only
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from traffic import analysis, db  # noqa: E402
from traffic.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path}")


def chart_hourly(series: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(series.index, series.values, marker=".", linewidth=1)  # NaN -> gap in the line
    ax.set(title="Преминавания на час (средно на камера; празно = няма данни)", xlabel="час", ylabel="брой/час")
    ax.grid(alpha=0.3)
    save(fig, path)


def chart_profile(series: pd.Series, path: Path) -> None:
    s = series.dropna()
    profile = s.groupby(s.index.hour).mean().reindex(range(24))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(profile.index, profile.values)
    ax.set(title="Среден дневен профил", xlabel="час от денонощието", ylabel="средно на час", xticks=range(0, 24, 2))
    ax.grid(axis="y", alpha=0.3)
    save(fig, path)


def chart_heatmap(grid: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(grid.to_numpy(dtype=float), aspect="auto", cmap="Blues")  # NaN cells stay blank
    ax.set(title="Натоварване: час × ден от седмицата (празно = няма данни)", xlabel="час от денонощието",
           xticks=range(24), yticks=range(7), yticklabels=WEEKDAYS)
    fig.colorbar(im, ax=ax, label="средно на час")
    save(fig, path)


def chart_classes(counts: pd.DataFrame, path: Path) -> None:
    table = counts.pivot_table(index="class_name", columns="direction", values="n", aggfunc="sum", fill_value=0)
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8, 4))
    table.plot.bar(ax=ax, rot=0)
    ax.set(title="Разпределение по клас и посока", xlabel="клас", ylabel="брой")
    ax.grid(axis="y", alpha=0.3)
    save(fig, path)


def chart_periods(periods: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(periods["период"], periods["средно на час"].fillna(0), color=["#f2b134", "#2c3e70", "#4f9d69", "#c0504d"])
    ax.set(title="Средно преминавания на час по периоди", ylabel="средно на час")
    ax.grid(axis="y", alpha=0.3)
    save(fig, path)


def chart_decomposition(dec: pd.DataFrame, found: pd.DataFrame, path: Path) -> None:
    parts = ["observed", "trend", "daily", "weekly", "residual"]
    titles = ["наблюдавано", "тенденция", "дневна сезонност", "седмична сезонност", "остатък"]
    fig, axes = plt.subplots(len(parts), 1, figsize=(11, 10), sharex=True)
    for ax, part, title in zip(axes, parts, titles, strict=True):
        ax.plot(dec.index, dec[part], linewidth=1)
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
    if not found.empty:
        axes[0].scatter(found.index, found["observed"], color="red", zorder=3, label="аномалия")
        axes[0].legend()
    axes[0].set_title("Декомпозиция на времевия ред")
    save(fig, path)


def parse_day(text: str | None, default: date) -> date:
    return date.fromisoformat(text) if text else default


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="first", help="first local day, YYYY-MM-DD (default: 6 days ago)")
    ap.add_argument("--to", dest="last", help="last local day, YYYY-MM-DD (default: today)")
    ap.add_argument("--camera", action="append", help="camera id (repeatable; default: all)")
    ap.add_argument("--day-hours", default=f"{analysis.DAY_HOURS[0]}-{analysis.DAY_HOURS[1]}", help="day period, e.g. 7-19")
    ap.add_argument("--out", type=Path, default=Path("reports"), help="output folder (default: reports)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = ap.parse_args(argv)

    config = load_config(args.config)
    tz = ZoneInfo(config.defaults.timezone)
    today = datetime.now(tz).date()
    last = parse_day(args.last, today)
    first = parse_day(args.first, last - timedelta(days=6))
    day_hours = tuple(int(x) for x in args.day_hours.split("-"))
    start = pd.Timestamp(datetime.combine(first, time.min, tz)).tz_convert("UTC")
    end = pd.Timestamp(datetime.combine(last + timedelta(days=1), time.min, tz)).tz_convert("UTC")

    with closing(db.connect(config.defaults.db_path, read_only=True)) as conn:
        w = analysis.load_window(conn, args.camera, start, end)
    cameras = list(w.coverage.columns)

    out = args.out / f"{first}_{last}"
    out.mkdir(parents=True, exist_ok=True)
    series = analysis.mean_rate(w.rates, tz)
    periods = analysis.compare_periods(series, day_hours)
    dec = analysis.decompose(series)
    found = analysis.anomalies(dec)
    gaps = analysis.outages(w.health, w.intervals, until=w.end, since=w.start)

    print(f"Writing to {out}/")
    totals = w.counts.groupby(["camera_id", "class_name", "direction"])["n"].sum().unstack("direction", fill_value=0)
    totals.to_csv(out / "totals_by_camera_class_direction.csv")
    cov = (analysis.covered_minutes_per_hour(w.coverage).sum() / 60).round(2).rename("operating_hours")
    cov.to_csv(out / "operating_hours_by_camera.csv")
    series.rename("crossings_per_hour_per_camera").to_csv(out / "hourly_rate.csv")
    periods.to_csv(out / "periods.csv", index=False)
    gaps.to_csv(out / "outages.csv", index=False)
    if dec is not None:
        dec.to_csv(out / "decomposition.csv")
        found.to_csv(out / "anomalies.csv")

    if w.counts.empty:
        print("  no crossings in this window - charts skipped")
    else:
        chart_hourly(series, out / "hourly.png")
        chart_profile(series, out / "daily_profile.png")
        chart_heatmap(analysis.heatmap(series), out / "heatmap.png")
        chart_classes(w.counts, out / "classes.png")
        chart_periods(periods, out / "periods.png")
        if dec is not None:
            chart_decomposition(dec, found, out / "decomposition.png")

    total = int(w.counts["n"].sum())
    lines = [
        f"Period: {first} .. {last} ({config.defaults.timezone}); cameras: {', '.join(cameras)}",
        f"Operating time: {w.covered_hours:.1f} camera-hours ({w.coverage_pct:.1f}% of the window)",
        f"Crossings: {total}" + (f"  ->  {total / w.covered_hours:.1f} per operating hour per camera"
                                 if w.covered_hours else ""),
        "",
        "Totals by camera / class / direction:",
        totals.to_string() if not totals.empty else "  (none)",
        "",
        "Operating hours by camera:",
        cov.to_string(),
        "",
        "Day / night, weekday / weekend (mean crossings per hour per camera):",
        periods.round(1).to_string(index=False),
        "",
        "Class shares, %:",
        analysis.class_shares(w.counts, tz, day_hours).to_string() if total else "  (none)",
        "",
        ("Decomposition: not enough data (need "
         f"{analysis.MIN_HOURS_DECOMPOSE} hours with >= {analysis.MIN_COVERED_MIN:g} operating minutes)")
        if dec is None else
        f"Decomposition: done (weekly component {'used' if dec.attrs['weekly_used'] else 'needs >= 14 days'}); "
        f"anomalies |z| >= {analysis.Z_THRESHOLD:g}: {len(found)}",
        "",
        f"Outages / no-data periods: {len(gaps)} (total {gaps['minutes'].sum():.0f} min)" if len(gaps) else "Outages: none",
    ]
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {out / 'summary.txt'}\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
