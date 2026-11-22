"""Pick the inference backend and export the YOLO weights to it once, before the cameras start.

On the AMD EPYC server (2026-11-23, 4 camera processes at once, yolo26n at 384x640):
PyTorch 8.6, OpenVINO 16 analysed frames/s per camera.

An exported model has a fixed input size, so the size is part of the export's name:
changing `imgsz` in config.yaml can never silently reuse an export made for another size.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path

from .config import Defaults

log = logging.getLogger(__name__)

Exporter = Callable[[str, str, int | tuple[int, int]], str]


def size_tag(imgsz: int | tuple[int, int]) -> str:
    h, w = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    return f"{h}x{w}"


def export_path(weights: str, backend: str, imgsz: int | tuple[int, int]) -> Path:
    """Where the export of `weights` for this backend and size lives (next to the weights)."""
    if backend != "openvino":
        raise ValueError(f"no export for backend {backend!r}")
    p = Path(weights)
    return p.with_name(f"{p.stem}_{size_tag(imgsz)}_openvino_model")


def ultralytics_export(weights: str, backend: str, imgsz: int | tuple[int, int]) -> str:
    from ultralytics import YOLO

    return str(YOLO(weights).export(format=backend, imgsz=imgsz))


def prepare_model(defaults: Defaults, exporter: Exporter = ultralytics_export) -> str:
    """Return the model path the camera processes should load.

    Exports on first use. Any failure falls back to the PyTorch weights, so a missing
    or broken backend makes the counter slower, never stops it.
    """
    if defaults.backend == "pytorch":
        return defaults.model
    target = export_path(defaults.model, defaults.backend, defaults.imgsz)
    if target.exists():
        return str(target)
    log.info("exporting %s to %s at %s (one time)...", defaults.model, defaults.backend, size_tag(defaults.imgsz))
    try:
        produced = Path(exporter(defaults.model, defaults.backend, defaults.imgsz))
        if produced.resolve() != target.resolve():
            shutil.rmtree(target, ignore_errors=True)
            produced.rename(target)
    except Exception as err:  # e.g. openvino not installed: run on PyTorch instead
        log.warning("export to %s failed (%s); using PyTorch weights %s", defaults.backend, err, defaults.model)
        return defaults.model
    log.info("model ready: %s", target)
    return str(target)
