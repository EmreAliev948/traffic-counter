"""Command line: python -m traffic run | dashboard | calibrate | manual-count | evaluate | report."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, ConfigError, load_config
from .worker import setup_logging, supervise

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ("calibrate", "manual-count", "evaluate", "report")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m traffic", description="Live traffic counter")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG_PATH), help="path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="count traffic on all enabled cameras")
    run.add_argument("--only", help="comma-separated camera ids (also runs disabled ones)")
    run.add_argument("--preview", action="store_true", help="show a live window (nothing is saved)")
    sub.add_parser("dashboard", help="open the Streamlit dashboard")
    for name in TOOLS:
        sub.add_parser(name, help=f"same as python -m tools.{name.replace('-', '_')}", add_help=False)

    args, rest = parser.parse_known_args(argv)
    if args.command in TOOLS:
        tool = "tools." + {"manual-count": "manual_counter"}.get(args.command, args.command)
        return subprocess.call([sys.executable, "-m", tool, "--config", args.config, *rest], cwd=ROOT)
    if rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    if args.command == "dashboard":
        app = Path(__file__).with_name("dashboard.py")
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--", "--config", args.config])

    setup_logging()
    try:
        config = load_config(args.config)
        supervise(config, args.only.split(",") if args.only else None, args.preview)
    except ConfigError as err:
        print(f"config error: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
