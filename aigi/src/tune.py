"""Tinh chỉnh siêu tham số tự động — CHỈ dựa trên tập VAL tách từ dữ liệu train (không bao giờ nhìn tập test).

Tiêu chí: AUROC trên họ mô phỏng giữ lại (train.val_holdout_env, mô phỏng "bộ sinh lạ"); nếu không có thì
AUROC trên 5% ảnh train giữ lại. Mọi cấu hình dùng cùng ngân sách (tune.epochs, tune.train_frac), cùng dữ liệu.

Thứ tự (mỗi bước dùng kết quả tốt nhất của bước trước):
  1. lr của thân chính (ERM, mọi nhóm đặc trưng)
  2. V-REx: beta x lam (giai đoạn chọn cổng)   -> 3. ngưỡng cổng (giai đoạn train gọn)
  4. IRM: beta (+ ngưỡng), GroupDRO: eta (+ ngưỡng), bản không cổng của IRM / GroupDRO
  5. lr cho baseline train lại: CNNSpot, UnivFD (CLIP linear probe), Aligned
Kết quả: <runs>/tuned.json — run_all.py tự đọc và dùng cho các run chính (3 seed, đủ dữ liệu, đủ epoch).

  python -m src.tune                 # lưới đầy đủ (~40 run ngắn)
  python -m src.tune --space quick   # lưới nhỏ (~15 run) để thử
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from .config import deepcopy, load_config
from .train import train_one
from .utils import get_logger, load_json, save_json

log = get_logger("tune")

SPACES = {
    "full": dict(lr=[5e-5, 1e-4, 2e-4], beta=[1, 10, 100], lam=[1e-4, 1e-3, 1e-2], thr=[0.3, 0.5, 0.7],
                 irm_beta=[1, 10, 100], dro_eta=[0.001, 0.01, 0.1], base_lr=[3e-5, 1e-4, 3e-4],
                 clip_lr=[1e-4, 1e-3, 1e-2]),
    "quick": dict(lr=[1e-4, 2e-4], beta=[1, 10], lam=[1e-3], thr=[0.5], irm_beta=[10], dro_eta=[0.01],
                  base_lr=[1e-4], clip_lr=[1e-3]),
}


def main_args(cfg) -> dict:
    """Thiết lập của phương pháp chính (config.yaml mục main)."""
    m = cfg.get("main", {})
    return dict(trunk=m.get("trunk", "timm:convnext_tiny.fb_in22k"), pretrained=m.get("pretrained", True),
                mode=m.get("mode", "b"), envs=m.get("envs", "all"))


def defaults() -> dict:
    """Giá trị dùng khi CHƯA chạy tinh chỉnh (để run_all vẫn chạy được)."""
    return dict(main_lr=1e-4, vrex=dict(beta=10, lam=1e-3, thr=0.5), irm=dict(beta=10, thr=0.5),
                irm_plain=dict(beta=10), dro=dict(eta=0.01, thr=0.5), dro_plain=dict(eta=0.01),
                lr=dict(cnnspot=1e-4, univfd=1e-3, aligned=1e-4))


def load_tuned(cfg) -> dict:
    p = Path(cfg["paths"]["runs"]) / "tuned.json"
    d = defaults()
    if p.exists():
        got = load_json(p)
        for k, v in got.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                d[k].update(v)
            else:
                d[k] = v
        d["_source"] = str(p)
    else:
        d["_source"] = "mặc định (chưa chạy src.tune)"
    return d


class Tuner:
    def __init__(self, cfg, space):
        self.base = deepcopy(cfg)
        tc = cfg.get("tune", {})
        self.root = Path(cfg["paths"]["runs"]) / "tune"
        self.base["paths"]["runs"] = str(self.root)
        t = self.base["train"]
        t["epochs"] = tc.get("epochs", 2)
        t["train_frac"] = tc.get("train_frac", 0.3)
        t["select_frac"] = 1.0           # train_frac đã giảm dữ liệu; không giảm thêm
        t["test_lists"] = []             # KHÔNG đánh giá test khi tinh chỉnh
        t["keep_final"] = False          # chỉ giữ result.json (tiết kiệm ổ)
        self.base["data_variant"] = cfg.get("main", {}).get("data", "raw")  # tinh chỉnh trên đúng dữ liệu train chính
        t["seed"] = 0
        if tc.get("steps_per_epoch"):
            t["steps_per_epoch"] = tc["steps_per_epoch"]
        self.S = SPACES[space]
        self.log = []

    def run(self, name, **kw):
        c = deepcopy(self.base)
        t = c["train"]
        sets = kw.pop("set", {})
        if "features" in kw:
            c["features"] = kw.pop("features")
        t.update(kw)
        for k, v in sets.items():
            t[k] = v
        try:
            res = train_one(c, name)
            sc = res.get("val_score", float("nan"))
        except (Exception, SystemExit) as e:  # một cấu hình lỗi không được làm hỏng cả bước tinh chỉnh
            log.error(f"[tune] {name} LỖI: {e}")
            sc = float("nan")
        self.log.append(dict(run=name, score=sc, **{k: v for k, v in kw.items() if k not in ("trunk",)}, **sets))
        log.info(f"[tune] {name}: VAL = {sc:.4f}")
        return sc

    @staticmethod
    def best(scored, fallback=None):
        ok = [(s, v) for s, v in scored if s == s]  # bỏ NaN
        if not ok:
            log.error("Không cấu hình nào có điểm VAL -> giữ giá trị mặc định. Kiểm tra val_lists / val_holdout_env.")
            return fallback if fallback is not None else scored[len(scored) // 2][1]
        return max(ok, key=lambda x: x[0])[1]


def tune(cfg, space="full", only=None):
    T, S, M = Tuner(cfg, space), SPACES[space], main_args(cfg)
    out = load_tuned(cfg); out.pop("_source", None)
    base = dict(trunk=M["trunk"], pretrained=M["pretrained"], mode=M["mode"], envs=M["envs"])
    want = lambda k: only is None or k in only
    p = Path(cfg["paths"]["runs"]) / "tuned.json"

    def checkpoint():  # lưu sau mỗi phần: lỗi giữa chừng vẫn giữ được phần đã chọn
        save_json(out, p)
        save_json(T.log, Path(cfg["paths"]["runs"]) / "tune_log.json")

    # 1. lr của thân chính (ERM, mọi nhóm)
    if want("lr"):
        out["main_lr"] = T.best([(T.run(f"lr_{lr:g}", **base, lr=lr, stage="final", beta=0.0, lam=0.0), lr)
                                 for lr in S["lr"]])
    checkpoint()
    lr = out["main_lr"]

    # 2-3. V-REx: beta x lam, rồi ngưỡng cổng
    if want("vrex"):
        sel = T.best([(T.run(f"vrex_b{b:g}_l{l:g}", **base, lr=lr, stage="select", penalty="vrex", beta=float(b), lam=l),
                       (b, l)) for b, l in itertools.product(S["beta"], S["lam"])])
        sel_run = f"vrex_b{sel[0]:g}_l{sel[1]:g}"
        thr = T.best([(T.run(f"vrex_final_t{th:g}", **base, lr=lr, stage="final", beta=0.0, lam=0.0,
                             keep_from=sel_run, gate_threshold=th), th) for th in S["thr"]])
        out["vrex"] = dict(beta=sel[0], lam=sel[1], thr=thr)
    checkpoint()
    lam = out["vrex"]["lam"]

    # 4. IRM / GroupDRO (cùng ngân sách, cùng cách chọn)
    if want("irm"):
        b = T.best([(T.run(f"irm_b{b:g}", **base, lr=lr, stage="select", penalty="irm", beta=float(b), lam=lam), b)
                    for b in S["irm_beta"]])
        th = T.best([(T.run(f"irm_final_t{th:g}", **base, lr=lr, stage="final", beta=0.0, lam=0.0,
                            keep_from=f"irm_b{b:g}", gate_threshold=th), th) for th in S["thr"]])
        out["irm"] = dict(beta=b, thr=th)
        out["irm_plain"] = dict(beta=T.best([(T.run(f"irmplain_b{bb:g}", **base, lr=lr, stage="final", penalty="irm",
                                                    beta=float(bb), lam=0.0), bb) for bb in S["irm_beta"]]))
    if want("dro"):
        e = T.best([(T.run(f"dro_e{e:g}", **base, lr=lr, stage="select", penalty="groupdro", beta=0.0, lam=lam,
                           set=dict(dro_eta=e)), e) for e in S["dro_eta"]])
        th = T.best([(T.run(f"dro_final_t{th:g}", **base, lr=lr, stage="final", beta=0.0, lam=0.0,
                            keep_from=f"dro_e{e:g}", gate_threshold=th), th) for th in S["thr"]])
        out["dro"] = dict(eta=e, thr=th)
        out["dro_plain"] = dict(eta=T.best([(T.run(f"droplain_e{ee:g}", **base, lr=lr, stage="final", penalty="groupdro",
                                                   beta=0.0, lam=0.0, set=dict(dro_eta=ee)), ee) for ee in S["dro_eta"]]))

    checkpoint()

    # 5. baseline train lại (cùng cách chọn lr)
    if want("baselines"):
        rgb = dict(mode="b", envs="none", stage="final", beta=0.0, lam=0.0)
        out["lr"] = dict(
            cnnspot=T.best([(T.run(f"cnnspot_lr{x:g}", **rgb, trunk=cfg.get("main", {}).get("baseline_trunk", "resnet50"), pretrained=M["pretrained"], lr=x,
                                   set=dict(aug_prob=0.1)), x) for x in S["base_lr"]]),
            univfd=T.best([(T.run(f"univfd_lr{x:g}", **rgb, trunk="clip_ft", lr=x,
                                  set=dict(clip_ft_blocks=0)), x) for x in S["clip_lr"]]),
            aligned=T.best([(T.run(f"aligned_lr{x:g}", trunk=cfg.get("main", {}).get("baseline_trunk", "resnet50"), pretrained=M["pretrained"], mode="a", envs=["vae_sd"],
                                   stage="final", beta=0.0, lam=0.0, lr=x), x) for x in S["base_lr"]]))

    checkpoint()
    log.info(f"Siêu tham số đã chọn (theo VAL) -> {p}: {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", choices=list(SPACES), default="full")
    ap.add_argument("--only", nargs="*", choices=["lr", "vrex", "irm", "dro", "baselines"])
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    tune(cfg, a.space, a.only)


if __name__ == "__main__":
    main()
