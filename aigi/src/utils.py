"""Tiện ích chung: thiết bị, seed, log, đọc/ghi danh sách ảnh."""
from __future__ import annotations

import csv
import json
import logging
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
LIST_FIELDS = ["path", "label", "env", "source", "split"]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    logger.addHandler(h)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)


def write_list(rows: list[dict], path: str | Path):
    """rows: dict có path,label,env,source,split."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LIST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in LIST_FIELDS})


def read_list(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
        r["env"] = int(r["env"]) if r.get("env", "") != "" else 0
    return rows


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=Path(__file__).resolve().parents[1]).decode().strip()
    except Exception:
        return "unknown"


def save_json(obj, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fp16_unsafe() -> bool:
    """GTX 16xx (TU116/TU117) hay cho NaN/ảnh đen với FP16 trong mô hình lớn (VAE, CLIP) -> dùng FP32.
    Ép bật/tắt bằng biến môi trường AIGI_FP32=1/0."""
    import os
    v = os.environ.get("AIGI_FP32")
    if v is not None:
        return v == "1"
    try:
        import torch
        return torch.cuda.is_available() and "GTX 16" in torch.cuda.get_device_name(0).upper()
    except Exception:
        return False
