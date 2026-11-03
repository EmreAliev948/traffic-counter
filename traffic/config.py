"""Load config.yaml into typed dataclasses and validate it with clear error messages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

Point = tuple[int, int]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class ConfigError(ValueError):
    """Raised when config.yaml is missing or invalid."""


@dataclass(frozen=True)
class Defaults:
    model: str = "yolo26n.pt"
    fallback_model: str = "yolo11n.pt"
    conf: float = 0.1
    imgsz: int = 640
    vid_stride: int = 2
    classes: tuple[int, ...] = (0, 1, 2, 3, 5, 7)
    db_path: Path = Path("data/traffic.db")
    timezone: str = "UTC"


@dataclass(frozen=True)
class Camera:
    id: str
    name: str
    url: str
    line: tuple[Point, Point]
    enabled: bool = True


@dataclass(frozen=True)
class Config:
    defaults: Defaults
    cameras: tuple[Camera, ...]

    def camera(self, camera_id: str) -> Camera:
        for cam in self.cameras:
            if cam.id == str(camera_id):
                return cam
        known = ", ".join(c.id for c in self.cameras)
        raise ConfigError(f"unknown camera '{camera_id}' (known: {known})")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as err:
        raise ConfigError(f"{path}: invalid YAML: {err}") from err
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping with 'defaults' and 'cameras'")
    defaults = _parse_defaults(raw.get("defaults") or {}, base_dir=path.parent)
    cameras = _parse_cameras(raw.get("cameras"))
    return Config(defaults=defaults, cameras=cameras)


def _parse_defaults(raw: dict[str, Any], base_dir: Path) -> Defaults:
    unknown = set(raw) - set(Defaults.__dataclass_fields__)
    if unknown:
        raise ConfigError(f"defaults: unknown key(s): {', '.join(sorted(unknown))}")
    d = Defaults()
    conf = float(raw.get("conf", d.conf))
    if not 0.0 < conf < 1.0:
        raise ConfigError(f"defaults.conf must be between 0 and 1, got {conf}")
    vid_stride = int(raw.get("vid_stride", d.vid_stride))
    if vid_stride < 1:
        raise ConfigError(f"defaults.vid_stride must be >= 1, got {vid_stride}")
    classes = raw.get("classes", list(d.classes))
    if not (isinstance(classes, list) and classes and all(isinstance(c, int) and c >= 0 for c in classes)):
        raise ConfigError(f"defaults.classes must be a non-empty list of COCO class ids, got {classes!r}")
    db_path = Path(raw.get("db_path", d.db_path))
    return Defaults(
        model=str(raw.get("model", d.model)),
        fallback_model=str(raw.get("fallback_model", d.fallback_model)),
        conf=conf,
        imgsz=int(raw.get("imgsz", d.imgsz)),
        vid_stride=vid_stride,
        classes=tuple(classes),
        db_path=db_path if db_path.is_absolute() else base_dir / db_path,
        timezone=str(raw.get("timezone", d.timezone)),
    )


def _parse_cameras(raw: Any) -> tuple[Camera, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError("'cameras' must be a non-empty list")
    cameras: list[Camera] = []
    for i, item in enumerate(raw):
        where = f"cameras[{i}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{where}: must be a mapping")
        for key in ("id", "url", "line"):
            if key not in item:
                raise ConfigError(f"{where}: missing required key '{key}'")
        cam_id = str(item["id"])
        cameras.append(Camera(
            id=cam_id,
            name=str(item.get("name") or f"camera {cam_id}"),
            url=str(item["url"]),
            line=_parse_line(item["line"], f"camera '{cam_id}'"),
            enabled=bool(item.get("enabled", True)),
        ))
    ids = [c.id for c in cameras]
    duplicates = sorted({x for x in ids if ids.count(x) > 1})
    if duplicates:
        raise ConfigError(f"duplicate camera id(s): {', '.join(duplicates)}")
    return tuple(cameras)


def _parse_line(raw: Any, where: str) -> tuple[Point, Point]:
    def is_point(p: Any) -> bool:
        return isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p)

    if not (isinstance(raw, list) and len(raw) == 2 and all(is_point(p) for p in raw)):
        raise ConfigError(f"{where}: 'line' must be [[x1, y1], [x2, y2]], got {raw!r}")
    a = (int(raw[0][0]), int(raw[0][1]))
    b = (int(raw[1][0]), int(raw[1][1]))
    if a == b:
        raise ConfigError(f"{where}: the two line points are identical")
    return a, b
