"""E2: đặc trưng thủ công + Logistic Regression (cách học đơn giản nhất, chạy được cả trên CPU/Mac).

  python -m src.train_lr --exp E2_lr_npr --features npr
  python -m src.train_lr --exp E2_lr_all --features rgb,fft,hh,wpt,npr,clip --n-train 20000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from .skfallback import LogisticRegression, make_pipeline, StandardScaler
from torch.utils.data import DataLoader

from .config import lists_dir, load_config
from .data import ImageListDataset, subsample
from .evaluate import per_source
from .features import ClipStore, handcrafted
from .train import ClipProvider
from .utils import get_device, get_logger, read_list, save_json


@torch.no_grad()
def extract(cfg, rows, feats, list_name, device):
    ds = ImageListDataset(rows, crop=cfg["train"]["crop"], train=False)
    dl = DataLoader(ds, batch_size=64, num_workers=cfg["train"]["num_workers"])
    clip = ClipProvider(cfg, [list_name], device, "clip" in feats)
    out = []
    for x, y, env, idx in dl:
        x = x.to(device)
        parts = [handcrafted(x, feats)]
        emb = clip(x, [rows[i]["path"] for i in idx.tolist()])
        if emb is not None:
            parts.append(emb)
        out.append(torch.cat(parts, 1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--data", choices=["raw", "debiased"], default="raw")
    ap.add_argument("--n-train", default="20000")
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    feats = a.features.split(",")
    run_dir = Path(cfg["paths"]["runs"]) / a.exp
    log = get_logger(f"lr.{a.exp}", run_dir / "train.log")
    if (run_dir / "result.json").exists():
        log.info("Đã có result.json, bỏ qua"); return
    device = get_device()
    name = cfg["train"]["train_list"] + ("_debiased" if a.data == "debiased" else "")
    rows = subsample(read_list(lists_dir(cfg) / f"{name}.csv"), a.n_train if a.n_train == "full" else int(a.n_train))
    X = extract(cfg, rows, feats, name, device)
    y = np.array([r["label"] for r in rows])
    log.info(f"Đặc trưng {feats}: {X.shape}")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    clf.fit(X, y)
    test = {}
    for tl in cfg["train"]["test_lists"]:
        p = lists_dir(cfg) / f"{tl}.csv"
        if not p.exists():
            continue
        trow = read_list(p)
        prob = clf.predict_proba(extract(cfg, trow, feats, tl, device))[:, 1]
        test[tl] = per_source(trow, prob)
        log.info(f"[{tl}] AUROC TB {test[tl].get('mean', test[tl]['all']).get('auroc', float('nan')):.4f}")
    save_json(dict(exp=a.exp, learner="logreg", features=feats, n_train=len(rows), dim=int(X.shape[1]), test=test),
              run_dir / "result.json")


if __name__ == "__main__":
    main()
