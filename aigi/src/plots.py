"""Hình cho paper (mục 9.4): bản đồ cổng, ma trận chuyển giao mô phỏng -> bộ sinh, đường cong độ bền.

  python -m src.plots gates --run N2_vrex
  python -m src.plots transfer --runs "N1_*" --list genimage_all_val
  python -m src.plots robust --runs N2_final_s0 E1_npr_raw_s0 --list genimage_all_val
"""
from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import load_config
from .utils import get_logger, load_json

log = get_logger("plots")


def _runs(cfg, pats):
    root = Path(cfg["paths"]["runs"])
    return sorted(p.name for p in root.iterdir() if (p / "result.json").exists()
                  and any(fnmatch.fnmatch(p.name, q) for q in pats))


def plot_gates(cfg, run, out):
    res = load_json(Path(cfg["paths"]["runs"]) / run / "result.json")
    hist = res.get("gate_hist") or [res["gates"]]
    names = list(hist[-1].keys())
    M = np.array([[h[n] for n in names] for h in hist])
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.35), 3 + 0.25 * len(hist)))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(hist)), [f"epoch {i}" for i in range(len(hist))])
    ax.set_title(f"Mức mở cổng theo epoch — {run}")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout(); fig.savefig(out, dpi=200); log.info(f"Lưu {out}")


def plot_transfer(cfg, pats, list_name, out):
    runs = _runs(cfg, pats)
    rows, cols = [], None
    for r in runs:
        t = load_json(Path(cfg["paths"]["runs"]) / r / "result.json")["test"].get(list_name, {})
        srcs = [s for s in t if s not in ("mean", "all")]
        cols = cols or srcs
        rows.append([t.get(s, {}).get("auroc", np.nan) for s in cols])
    M = np.array(rows)
    fig, ax = plt.subplots(figsize=(max(6, 2 + 0.8 * len(cols)), max(3, 1 + 0.5 * len(runs))))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.5, vmax=1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j] * 100:.0f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right"); ax.set_yticks(range(len(runs)), runs)
    ax.set_title("AUROC (%): họ mô phỏng (hàng) -> bộ sinh thật (cột)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(out, dpi=200); log.info(f"Lưu {out}")


def plot_robust(cfg, runs, list_name, out):
    kinds = {"jpeg": "JPEG q", "resize": "resize x", "blur": "blur σ"}
    fig, axes = plt.subplots(1, len(kinds), figsize=(4 * len(kinds), 3.2))
    for ax, (k, lab) in zip(axes, kinds.items()):
        levels = cfg["eval"]["robust"][k]
        for r in runs:
            rd = Path(cfg["paths"]["runs"]) / r
            ys = []
            for lv in levels:
                f = rd / f"eval_{k}{lv}.json"
                ys.append(load_json(f)[list_name]["mean"]["auroc"] if f.exists() else np.nan)
            ax.plot([str(l) for l in levels], ys, marker="o", label=r)
        ax.set_xlabel(lab); ax.set_ylabel("AUROC TB"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=200); log.info(f"Lưu {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gates", "transfer", "robust"])
    ap.add_argument("--run")
    ap.add_argument("--runs", nargs="*", default=["*"])
    ap.add_argument("--list", default="genimage_all_val")
    ap.add_argument("--out", default=None)
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    Path("paper/figures").mkdir(parents=True, exist_ok=True)
    if a.cmd == "gates":
        plot_gates(cfg, a.run, a.out or f"paper/figures/gates_{a.run}.png")
    elif a.cmd == "transfer":
        plot_transfer(cfg, a.runs, a.list, a.out or f"paper/figures/transfer_{a.list}.png")
    else:
        plot_robust(cfg, a.runs, a.list, a.out or f"paper/figures/robust_{a.list}.png")


if __name__ == "__main__":
    main()
