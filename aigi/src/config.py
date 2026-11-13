"""Đọc config.yaml và ghi đè bằng --set khoa.con=gia_tri."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_value(v: str):
    # YAML 1.1 đọc "1e-3" thành chuỗi -> thử số trước
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    try:
        return yaml.safe_load(v)
    except Exception:
        return v


def set_by_path(cfg: dict, dotted: str, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def get_by_path(cfg: dict, dotted: str, default=None):
    d = cfg
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def deep_merge(base: dict, extra: dict) -> dict:
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """config.yaml + config.local.yaml (nếu có, do src.prepare_data tự ghi: đường dẫn dữ liệu trên máy bạn)
    + tuned.yaml (nếu có, do src.tune ghi: siêu tham số đã chọn trên tập val) + --set."""
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for extra in (path.with_name(path.stem + ".local.yaml"),):
        if extra.exists():
            deep_merge(cfg, yaml.safe_load(extra.read_text(encoding="utf-8")) or {})
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set cần dạng khoa=gia_tri, nhận được: {item}")
        k, v = item.split("=", 1)
        set_by_path(cfg, k.strip(), _parse_value(v.strip()))
    return cfg


def lists_dir(cfg: dict) -> Path:
    p = Path(cfg["paths"]["lists"])
    return p if p.is_absolute() else PROJECT_ROOT / p


def dump(cfg: dict, path: str | Path):
    Path(path).write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def deepcopy(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
