"""Give each camera process its own physical CPU core(s) (Linux only).

Without this, OpenVINO in every camera process pins its threads to the same logical CPUs
(one per physical core), so 4 cameras crowded onto CPUs 4-7 while 0-3 idled. Measured on the
AMD EPYC server (4 cores x 2 hyperthreads, 4 cameras, 2026-11-23):
    default 63.5, own core per camera 77.4 analysed frames/s in total (+22 %).
The camera process restricts itself to its CPUs before the model loads, so OpenVINO and
PyTorch only ever see those CPUs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

SYS_CPU = Path("/sys/devices/system/cpu")


def _parse_cpu_list(text: str) -> set[int]:
    """'0,4' or '0-1,4-5' -> {0, 1, 4, 5}"""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def physical_cores(allowed: set[int], sys_cpu: Path = SYS_CPU) -> list[set[int]]:
    """Group the allowed logical CPUs by physical core (hyperthread siblings together)."""
    cores: list[set[int]] = []
    for cpu in sorted(allowed):
        try:
            siblings = _parse_cpu_list((sys_cpu / f"cpu{cpu}/topology/thread_siblings_list").read_text())
        except (OSError, ValueError):
            siblings = {cpu}
        group = (siblings & allowed) or {cpu}
        if group not in cores:
            cores.append(group)
    return cores


def assign_cores(cores: list[set[int]], n_cameras: int) -> list[set[int]]:
    """Share the cores out: each camera gets whole cores, as evenly as possible.

    More cameras than cores: cameras share cores round-robin.
    """
    if n_cameras <= 0 or not cores:
        return []
    if n_cameras >= len(cores):
        return [set(cores[i % len(cores)]) for i in range(n_cameras)]
    per, extra = divmod(len(cores), n_cameras)
    groups, start = [], 0
    for i in range(n_cameras):
        size = per + (i < extra)
        groups.append(set().union(*cores[start:start + size]))
        start += size
    return groups


def plan(n_cameras: int, enabled: bool = True) -> list[set[int] | None]:
    """CPUs for each camera, or None everywhere when pinning is off or not possible here."""
    if not enabled or not hasattr(os, "sched_getaffinity"):
        return [None] * n_cameras
    groups = assign_cores(physical_cores(set(os.sched_getaffinity(0))), n_cameras)
    return groups or [None] * n_cameras


def pin_current_process(cpus: set[int] | None) -> None:
    if cpus is None:
        return
    try:
        os.sched_setaffinity(0, cpus)
    except OSError as err:
        log.warning("cannot pin to CPUs %s: %s", sorted(cpus), err)
