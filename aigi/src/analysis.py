"""Phân tích cho phần đóng góp (mục 9.x tài liệu).

1) Độ ổn định của cổng qua nhiều seed: các lần chọn có ra cùng tập nhóm không?
   python -m src.analysis gate-stability --runs N4_b_vrex STAB_N4_vrex_s1 STAB_N4_vrex_s2
2) Bất biến trên ảnh MÔ PHỎNG có dự đoán được khả năng CHUYỂN GIAO sang bộ sinh THẬT không?
   Với từng nhóm đặc trưng: train một bộ phân loại tuyến tính (LR) chỉ trên dữ liệu mô phỏng, đo
     - inv_std  : độ lệch chuẩn AUROC giữa các môi trường mô phỏng (thấp = bất biến)
     - sim_auc  : AUROC trung bình trên mô phỏng (độ chính xác trong miền)
     - gate     : mức mở cổng do phương pháp học (run select chế độ a, không thấy ảnh sinh thật)
     - transfer : AUROC trên bộ sinh thật CHƯA có trong mô phỏng (unseen) và trên tất cả
   rồi tính tương quan hạng Spearman giữa (−inv_std, gate, sim_auc) và transfer trên 27 nhóm.
   Giả thuyết của bài: −inv_std và gate tương quan với transfer MẠNH HƠN sim_auc.
   python -m src.analysis invariance-transfer --gate-run N2_vrex --target genimage_all_val
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import lists_dir, load_config
from .data import ImageListDataset
from .evaluate import overlap_group, sample_rows
from .features import compute_groups, group_spec
from .utils import get_device, get_logger, load_json, read_list, save_json

log = get_logger("analysis")


# ---------------------------------------------------------------- 1) ổn định cổng
def cmd_gate_stability(cfg, runs, thr):
    from .skfallback import spearmanr
    root = Path(cfg["paths"]["runs"])
    gates = {}
    for r in runs:
        f = root / r / "result.json"
        if f.exists() and load_json(f).get("gates"):
            gates[r] = load_json(f)["gates"]
        else:
            log.warning(f"{r}: chưa có result.json có cổng -> bỏ qua")
    if len(gates) < 2:
        raise SystemExit("Cần ít nhất 2 run select có cổng")
    names = sorted(set.intersection(*[set(g) for g in gates.values()]))
    keep = {r: {n for n in names if g[n] > thr} for r, g in gates.items()}
    pairs = []
    for a, b in itertools.combinations(gates, 2):
        inter, union = keep[a] & keep[b], keep[a] | keep[b]
        rho = spearmanr([gates[a][n] for n in names], [gates[b][n] for n in names]).statistic
        pairs.append(dict(a=a, b=b, jaccard=len(inter) / max(1, len(union)), spearman=float(rho)))
    freq = {n: sum(n in k for k in keep.values()) / len(keep) for n in names}
    res = dict(runs=list(gates), threshold=thr, pairs=pairs,
               mean_jaccard=float(np.mean([p["jaccard"] for p in pairs])),
               mean_spearman=float(np.mean([p["spearman"] for p in pairs])),
               keep_frequency=dict(sorted(freq.items(), key=lambda kv: -kv[1])),
               always_kept=[n for n, f in freq.items() if f == 1.0])
    out = Path(cfg["paths"]["runs"]) / "analysis_gate_stability.json"
    save_json(res, out)
    log.info(f"Jaccard TB {res['mean_jaccard']:.3f} | Spearman TB {res['mean_spearman']:.3f} | "
             f"luôn được giữ: {res['always_kept']} -> {out}")


# ---------------------------------------------------------------- 2) bất biến -> chuyển giao
def _descriptors(x, spec):
    """Mô tả thống kê gọn của từng nhóm: kênh không gian -> (mean, std, mean|.|, kurtosis); vector -> giữ nguyên."""
    sp, vec = compute_groups(x, spec, None)
    out = []
    for t in sp:
        f = t.flatten(2)
        m, s = f.mean(-1), f.std(-1) + 1e-6
        kurt = (((f - m[..., None]) / s[..., None]) ** 4).mean(-1)
        out.append(torch.cat([m, s, f.abs().mean(-1), kurt.clamp(max=50)], 1))
    out += vec
    return out


@torch.no_grad()
def _extract(cfg, rows, spec, device):
    ds = ImageListDataset(rows, crop=cfg["eval"]["crop"], train=False)
    dl = DataLoader(ds, batch_size=cfg["eval"]["batch_size"], num_workers=cfg["train"]["num_workers"])
    feats = [[] for _ in spec]
    for x, *_ in dl:
        for k, d in enumerate(_descriptors(x.to(device), spec)):
            feats[k].append(d.float().cpu().numpy())
    return [np.concatenate(f) for f in feats]


def cmd_invariance_transfer(cfg, gate_run, target, n_per, n_target):
    from .skfallback import spearmanr
    from .skfallback import LogisticRegression, roc_auc_score, make_pipeline, StandardScaler
    device = get_device()
    feats = [f for f in cfg["features"] if f != "clip"]  # CLIP không có mô tả thống kê theo kênh -> bỏ qua ở phân tích này
    spec = group_spec(feats, cfg["clip"]["arch"])
    rng = np.random.default_rng(0)
    # dữ liệu mô phỏng: mỗi họ là một môi trường, chia đôi train/held-out
    env_rows = {}
    for fam in cfg["sim"]["families"]:
        p = lists_dir(cfg) / f"sim_{fam}.csv"
        if p.exists():
            env_rows[fam] = sample_rows(read_list(p), n_per, seed=1)
    if len(env_rows) < 2:
        raise SystemExit("Cần ít nhất 2 họ mô phỏng (đợt 2)")
    tgt = sample_rows(read_list(lists_dir(cfg) / f"{target}.csv"), n_target, seed=2)
    log.info(f"Mô phỏng {[(k, len(v)) for k, v in env_rows.items()]} | đích {target}: {len(tgt)} ảnh | {len(spec)} nhóm")
    Xe = {k: _extract(cfg, v, spec, device) for k, v in env_rows.items()}
    ye = {k: np.array([r["label"] for r in v]) for k, v in env_rows.items()}
    split = {k: rng.random(len(v)) < 0.5 for k, v in env_rows.items()}
    Xt = _extract(cfg, tgt, spec, device)
    yt = np.array([r["label"] for r in tgt]); st = np.array([r["source"] for r in tgt])
    overlap = cfg["eval"].get("sim_overlap")
    grp = {s: overlap_group(s, overlap) for s in set(st)}
    gates = load_json(Path(cfg["paths"]["runs"]) / gate_run / "result.json").get("gates", {}) if gate_run else {}

    def auc_by_source(p, which):
        vals = []
        for s in sorted(set(st)):
            if which != "all" and grp[s] != which:
                continue
            m = st == s
            if not (yt[m] == 1).any():
                continue
            if not (yt[m] == 0).any():
                m = m | (yt == 0)
            vals.append(roc_auc_score(yt[m], p[m]))
        return float(np.mean(vals)) if vals else float("nan")

    rows = []
    for k, g in enumerate(spec):
        Xtr = np.concatenate([Xe[e][k][split[e]] for e in Xe]); ytr = np.concatenate([ye[e][split[e]] for e in Xe])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(Xtr, ytr)
        env_auc = [roc_auc_score(ye[e][~split[e]], clf.predict_proba(Xe[e][k][~split[e]])[:, 1]) for e in Xe]
        pt = clf.predict_proba(Xt[k])[:, 1]
        rows.append(dict(group=g.name, gate=gates.get(g.name, float("nan")), inv_std=float(np.std(env_auc)),
                         sim_auc=float(np.mean(env_auc)), transfer_all=auc_by_source(pt, "all"),
                         transfer_unseen=auc_by_source(pt, "unseen"), transfer_seen=auc_by_source(pt, "seen")))
        log.info(f"  {g.name:12} inv_std {rows[-1]['inv_std']:.3f} sim {rows[-1]['sim_auc']:.3f} "
                 f"-> unseen {rows[-1]['transfer_unseen']:.3f}")

    def corr(a, b):
        x = np.array([r[a] for r in rows], float) * (-1 if a == "inv_std" else 1)
        y = np.array([r[b] for r in rows], float)
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < 4:
            return None
        res = spearmanr(x[ok], y[ok])
        return dict(rho=float(res.statistic), p=float(res.pvalue), n=int(ok.sum()))

    target_key = "transfer_unseen" if any(grp[s] == "unseen" for s in grp) else "transfer_all"
    summary = {f"{'-inv_std' if a == 'inv_std' else a} vs {target_key}": corr(a, target_key)
               for a in ("inv_std", "gate", "sim_auc")}
    out_dir = Path(cfg["paths"]["runs"])
    save_json(dict(target=target, gate_run=gate_run, groups=rows, spearman=summary,
                   note="Dấu: -inv_std để 'bất biến hơn' = giá trị lớn hơn."), out_dir / "analysis_invariance_transfer.json")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
        for a_, (key, lab) in zip(ax, [("inv_std", "Độ lệch AUROC giữa môi trường mô phỏng (thấp = bất biến)"),
                                       ("sim_auc", "AUROC trên mô phỏng (trong miền)")]):
            a_.scatter([r[key] for r in rows], [r[target_key] for r in rows], s=18)
            for r in rows:
                a_.annotate(r["group"], (r[key], r[target_key]), fontsize=6)
            a_.set_xlabel(lab, fontsize=8); a_.set_ylabel(f"AUROC {target_key}", fontsize=8)
        fig.tight_layout()
        Path("paper/figures").mkdir(parents=True, exist_ok=True)
        fig.savefig("paper/figures/invariance_transfer.png", dpi=200)
    except Exception as e:
        log.warning(f"Không vẽ được hình: {e}")
    log.info(f"Spearman: {summary}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate-stability", "invariance-transfer"])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--gate-run")
    ap.add_argument("--target", default="genimage_all_val")
    ap.add_argument("--n-per", type=int, default=500, help="ảnh mỗi (họ mô phỏng, nhãn)")
    ap.add_argument("--n-target", type=int, default=300, help="ảnh mỗi (bộ sinh đích, nhãn)")
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    if a.cmd == "gate-stability":
        cmd_gate_stability(cfg, a.runs, a.threshold if a.threshold is not None else cfg["train"]["gate_threshold"])
    else:
        cmd_invariance_transfer(cfg, a.gate_run, a.target, a.n_per, a.n_target)


if __name__ == "__main__":
    main()
