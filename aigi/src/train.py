"""Train (mục 8): ERM hoặc V-REx + cổng, hai giai đoạn (select -> final), thân CNN/ViT.

Ví dụ:
  # E1: chỉ NPR, ERM thường, dữ liệu gốc
  python -m src.train --exp E1_npr_raw --features npr --data raw
  # N2: chọn đặc trưng bất biến trên môi trường mô phỏng (giai đoạn 1)
  python -m src.train --exp N2_vrex --mode a --envs all --beta 10 --lam 1e-3 --stage select
  # N2: train gọn với nhóm được giữ (giai đoạn 2), 3 seed
  python -m src.train --exp N2_final --mode a --envs all --stage final --keep-from N2_vrex --seeds 0 1 2
  # V1: thân ViT scratch trên NPR
  python -m src.train --exp V1_vit_npr --trunk vit --features npr
  # V2: ViT pretrained trên ảnh RGB
  python -m src.train --exp V2_vit_rgb --trunk vit_rgb --pretrained 1
  # Mô hình mạnh từ khảo sát (mục 8.6): ConvNeXt, DINOv2, CLIP fine-tune, ResNet-50
  python -m src.train --exp S_convnext --trunk convnext --pretrained 1 --set train.lr=1e-4
  python -m src.train --exp S_dinov2_lp --trunk dinov2 --pretrained 1 --set train.freeze_backbone=true train.lr=1e-3
  python -m src.train --exp S_clip_ft --trunk clip_ft --set train.clip_ft_blocks=2 train.lr=2e-5
  # Phương pháp chính với thân mạnh: cổng + V-REx trên thân ConvNeXt pretrained
  python -m src.train --exp S_gated_convnext --trunk timm:convnext_tiny.fb_in22k --pretrained 1 --mode b --envs all --beta 10 --lam 1e-3 --stage select
Mỗi run ghi <runs>/<exp>/{config.json, train.log, last.pt, final.pt, result.json}.
Chạy lại cùng lệnh: bỏ qua run đã có result.json, tiếp tục từ last.pt nếu bị ngắt.
"""
from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import deepcopy, dump, lists_dir, load_config
from .data import CropCache, ImageListDataset, collate_full, collate_tiles, subsample
from .features import ClipEncoder, ClipStore, FullClipCache, group_spec
from .models import build_model
from .utils import get_device, get_logger, git_commit, load_json, read_list, save_json, seed_all


# ---------------------------------------------------------------- dữ liệu train
def resolve_envs(cfg):
    t = cfg["train"]
    if t["envs"] in (None, "none", [], "[]"):
        return []
    fams = cfg["sim"]["families"] if t["envs"] == "all" else list(t["envs"])
    hold = t.get("val_holdout_env")
    return [f for f in fams if f != hold]  # họ mô phỏng giữ lại làm VAL thì không được train


def build_train_rows(cfg, log):
    """Trả về (rows, env_names, list_names_dùng). env id = vị trí trong env_names."""
    t = cfg["train"]
    rows, env_names, used = [], [], []
    for fam in resolve_envs(cfg):
        p = lists_dir(cfg) / f"sim_{fam}{t.get('env_suffix') or ''}.csv"  # env_suffix: bản rút gọn (profile vitnhanh)
        if not p.exists():
            log.warning(f"Thiếu {p.name} — chạy: python -m src.simulate --family {fam}")
            continue
        e = len(env_names)
        env_names.append(fam)
        rows += [dict(r, env=e) for r in read_list(p)]
        used.append(p.stem)
    if t["mode"] == "b":
        name = t["train_list"] + ("_debiased" if cfg.get("data_variant") == "debiased" else "")
        p = lists_dir(cfg) / f"{name}.csv"
        gen = subsample(read_list(p), t["n_train"], seed=t["seed"])
        e = len(env_names)
        env_names.append(f"gen:{t['train_list']}")
        rows += [dict(r, env=e) for r in gen]
        used.append(name)
    if not rows:
        hold = t.get("val_holdout_env")
        raise SystemExit("Không có dữ liệu train: kiểm tra mode/envs và các file trong lists/"
                         + (f" (lưu ý: họ '{hold}' đang được giữ lại làm VAL nên không dùng để train)" if hold else ""))
    frac = float(t.get("select_frac", 1.0)) if t["stage"] == "select" else 1.0
    frac *= float(t.get("train_frac", 1.0))  # tinh chỉnh tham số dùng một phần dữ liệu (như nhau cho mọi cấu hình)
    if frac < 1.0:
        # Giai đoạn chọn cổng chạy trên một phần dữ liệu (cùng tỉ lệ cho MỌI run select để công bằng);
        # giai đoạn final luôn dùng đủ dữ liệu.
        rng = np.random.default_rng(t["seed"])
        keep = []
        for key in sorted({(r["env"], r["label"]) for r in rows}):
            idx = [i for i, r in enumerate(rows) if (r["env"], r["label"]) == key]
            keep += list(rng.choice(idx, max(1, int(len(idx) * frac)), replace=False))
        rows = [rows[i] for i in sorted(keep)]
        log.info(f"Tỉ lệ dữ liệu {frac:.2f}: dùng {len(rows)} ảnh")
    log.info(f"Train: {len(rows)} ảnh | môi trường {env_names} | "
             f"{dict(Counter((r['env'], r['label']) for r in rows))}")
    return rows, env_names, used


def balanced_sampler(rows, num_samples):
    cnt = Counter((r["env"], r["label"]) for r in rows)
    w = torch.tensor([1.0 / cnt[(r["env"], r["label"])] for r in rows], dtype=torch.double)
    return WeightedRandomSampler(w, num_samples=num_samples, replacement=True)


# ---------------------------------------------------------------- CLIP
class ClipProvider:
    def __init__(self, cfg, list_names, device, needed):
        self.needed, self.device, self.cfg = needed, device, cfg
        self.store = ClipStore(cfg, list_names) if needed else None
        self.enc = None

    def __call__(self, x, paths):
        if not self.needed:
            return None
        emb, mask = self.store.lookup(paths)
        if emb is None:
            emb = torch.zeros(len(paths), 0)
        if not mask.all():
            if self.enc is None:
                self.enc = ClipEncoder(self.cfg["clip"]["arch"], self.cfg["clip"]["pretrained"], self.device)
            fresh = self.enc(x[~mask])
            full = torch.zeros(len(paths), fresh.shape[1])
            if mask.any():
                full[mask] = emb[mask]
            full[~mask] = fresh.cpu()
            emb = full
        return emb.to(self.device)


# ---------------------------------------------------------------- VAL (chọn checkpoint / siêu tham số)
def build_val_sets(cfg, log):
    """Trả về {tên: rows}. Chỉ dùng dữ liệu tách từ TRAIN; KHÔNG dùng tập test.
    - 'indomain': genimage_<train>_trainval (5% ảnh train giữ lại)
    - 'holdout' : họ mô phỏng train.val_holdout_env (không có trong môi trường train) — mô phỏng 'bộ sinh lạ'."""
    from .evaluate import sample_rows
    t, out = cfg["train"], {}
    n = t.get("val_max_per_source", 1000)
    for name in t.get("val_lists", []) or []:
        if cfg.get("data_variant") == "debiased" and (lists_dir(cfg) / f"{name}_debiased.csv").exists():
            name = f"{name}_debiased"  # train trên dữ liệu sạch -> VAL cũng sạch
        p = lists_dir(cfg) / f"{name}.csv"
        if p.exists():
            out["indomain"] = (name, sample_rows(read_list(p), n, seed=0))
    hold = t.get("val_holdout_env")
    if hold:
        p = lists_dir(cfg) / f"sim_{hold}.csv"
        if p.exists():
            out["holdout"] = (f"sim_{hold}", sample_rows(read_list(p), n, seed=0))
        else:
            log.warning(f"Thiếu sim_{hold}.csv cho VAL holdout")
    return out


def validate(model, cfg, val_sets):
    from .skfallback import roc_auc_score
    from .evaluate import predict
    res = {}
    for key, (name, rows) in val_sets.items():
        y = np.array([r["label"] for r in rows])
        if len(set(y)) < 2:
            continue
        p = predict(model, cfg, rows, clip_lists=[name])
        res[key] = float(roc_auc_score(y, p))
    return res


def val_score(vres, t):
    """Tiêu chí chọn: holdout (tổng quát hoá) nếu có, không thì indomain."""
    key = t.get("select_by", "auto")
    if key == "auto":
        key = "holdout" if "holdout" in vres else "indomain"
    return vres.get(key, float("nan")), key


# ---------------------------------------------------------------- train 1 run
def train_one(cfg, exp: str, force=False):
    t = cfg["train"]
    run_dir = Path(cfg["paths"]["runs"]) / exp
    run_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"train.{exp}", run_dir / "train.log")
    if (run_dir / "result.json").exists() and not force:
        log.info(f"{exp}: đã có result.json -> bỏ qua (dùng --force để chạy lại)")
        return load_json(run_dir / "result.json")
    seed_all(t["seed"])
    device = get_device()
    log.info(f"Thiết bị: {device} | commit {git_commit()}")

    spec = group_spec(cfg["features"], cfg["clip"]["arch"])
    if t["stage"] == "final" and t.get("keep_from"):
        res = load_json(Path(cfg["paths"]["runs"]) / t["keep_from"] / "result.json")
        keep = {k for k, v in res["gates"].items() if v > t["gate_threshold"]}
        if t.get("keep_random"):
            # Đối chứng: chọn NGẪU NHIÊN cùng số nhóm -> chứng minh cổng chọn đúng chứ không nhờ ít nhóm hơn
            rng = np.random.default_rng(1000 + t["seed"])
            keep = set(rng.choice([g.name for g in spec], len(keep), replace=False).tolist())
            log.info("keep_random: chọn ngẫu nhiên cùng số nhóm")
        spec = [g for g in spec if g.name in keep]
        log.info(f"Giữ {len(spec)} nhóm từ {t['keep_from']}: {[g.name for g in spec]}")
        if not spec:
            raise SystemExit("Không còn nhóm nào sau khi lọc cổng — giảm gate_threshold hoặc lam")
    use_gates = t["stage"] == "select"
    model = build_model(t, spec, use_gates, cfg["clip"]).to(device)
    if not hasattr(model, "sp_groups"):  # mô hình trên ảnh RGB, không dùng nhóm đặc trưng
        spec = []
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Mô hình {t['trunk']} | {n_params / 1e6:.2f}M tham số train được | "
             f"nhóm: {[g.name for g in spec] or 'ảnh RGB (không cổng)'}")

    rows, env_names, used = build_train_rows(cfg, log)
    n_env = len(env_names)
    # full_image_crop: mỗi epoch cắt NGẪU NHIÊN 1 vùng crop x crop ở bất kỳ đâu trong TOÀN ẢNH (không resize)
    # -> mô hình thấy nhiều vùng khác nhau của mỗi ảnh. Cache crop chỉ lưu vùng giữa nên phải tắt cache.
    milc = t.get("mil") or {}
    mil = bool(getattr(model, "is_mil", False))
    full_img = bool(getattr(model, "is_full", False))
    K = int(milc.get("windows_train" if full_img else "tiles_train", 4)) if mil else 0
    full = t.get("full_image_crop", False) or mil
    cache = CropCache(cfg, used) if cfg.get("cache", {}).get("use", True) and not full else None
    if full and not mil:
        log.info("full_image_crop: cắt ngẫu nhiên trên toàn ảnh mỗi epoch (không dùng cache vùng giữa)")
    if cache is not None and len(cache):
        log.info(f"Dùng cache crop: {len(cache)} ảnh (không mất dữ liệu)")
    ds = ImageListDataset(rows, crop=t["crop"], train=True, aug_prob=t["aug_prob"], cache=cache,
                          train_tiles=0 if full_img else K, mode="full" if full_img else "center")
    # MIL: batch_size tính theo SỐ MẢNH -> mỗi bước lấy batch_size // K ảnh, mỗi ảnh tối đa K mảnh (vừa GPU 4 GB)
    bsz = max(1, t["batch_size"] // K) if mil else t["batch_size"]
    if full_img:
        log.info(f"Toàn ảnh (không cắt): đặc trưng tính trên cả ảnh, cửa sổ 256x256 trượt bước {milc.get('stride', 128)} px, "
                 f"mỗi ảnh tối đa {K} cửa sổ/bước khi train, {bsz} ảnh/bước, FFT + CLIP trên cả ảnh, gộp bằng attention")
    elif mil:
        log.info(f"MIL: mỗi ảnh tối đa {K} mảnh ngẫu nhiên trên toàn ảnh, {bsz} ảnh/bước, gộp bằng attention")
    steps = t["steps_per_epoch"] or math.ceil(len(ds) / bsz)
    dl = DataLoader(ds, batch_size=bsz, sampler=balanced_sampler(rows, steps * bsz),
                    collate_fn=collate_full if full_img else collate_tiles if mil else None,
                    num_workers=t["num_workers"], pin_memory=device.type == "cuda", drop_last=True,
                    persistent_workers=t["num_workers"] > 0,
                    prefetch_factor=4 if t["num_workers"] > 0 else None)
    clip = ClipProvider(cfg, used, device, getattr(model, "needs_clip", False))
    fclip = FullClipCache(cfg, device, full_img and getattr(model, "needs_clip", False))

    # Nhóm tham số: thân (lr gốc) / stem+head (lr x head_lr_mult, mặc định 1) / cổng (lr x10, không weight decay).
    # Lớp mới khởi tạo ngẫu nhiên cần lr lớn hơn thân đã học sẵn (cách fine-tune chuẩn của ConvNeXt/ViT).
    mult = float(t.get("head_lr_mult", 1.0))
    _n = lambda n: n[5:] if n.startswith("base.") else n  # MIL bọc mô hình trong .base
    is_trunk = lambda n: _n(n).startswith(("trunk.", "backbone.", "model.", "clip.", "m.", "visual."))
    is_gate = lambda n: _n(n) == "gate"
    trunk_p = [p for n, p in model.named_parameters() if p.requires_grad and not is_gate(n) and is_trunk(n)]
    head_p = [p for n, p in model.named_parameters() if p.requires_grad and not is_gate(n) and not is_trunk(n)]
    groups = [{"params": trunk_p, "weight_decay": t["weight_decay"]}] if trunk_p else []
    if head_p:
        groups.append({"params": head_p, "weight_decay": t["weight_decay"], "lr": t["lr"] * mult})
    if use_gates:
        gm = float(t.get("gate_lr_mult", 10.0))  # cổng cần lr lớn: AdamW dịch alpha ~lr/bước, từ 0.88 xuống 0.5 cần alpha giảm 2
        groups.append({"params": [model.gate], "weight_decay": 0.0, "lr": t["lr"] * gm})
    opt = None
    if t.get("optim", "adamw") == "adamw8bit" and device.type == "cuda":
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(groups, lr=t["lr"])
            log.info("Optimizer AdamW 8-bit (bitsandbytes)")
        except Exception as e:
            log.warning(f"Không dùng được AdamW 8-bit ({e}) -> AdamW thường")
    if opt is None:
        opt = torch.optim.AdamW(groups, lr=t["lr"])
    total = t["epochs"] * steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g.get("lr", t["lr"]) for g in groups],
                                                total_steps=total, pct_start=min(0.3, 1 / max(1, t["epochs"])))
    use_amp = t["amp"] and device.type == "cuda"
    prec = t.get("precision", "fp16")
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(0)
        if prec == "bf16" and cap < (8, 0):  # trước Ampere bf16 chỉ được GIẢ LẬP -> rất chậm
            prec = "fp16"
        from .utils import fp16_unsafe
        if prec == "fp16" and fp16_unsafe():  # GTX 16xx: fp16 chậm/NaN -> fp32
            prec = "fp32"
    log.info(f"Độ chính xác khi train: {prec}")
    amp_dtype = torch.bfloat16 if prec == "bf16" else torch.float16
    use_amp = use_amp and prec != "fp32"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    if t.get("channels_last") and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    # EMA trọng số (train.ema = 0.999...; 0 = tắt): trung bình trượt của tham số, thường tổng quát hoá tốt hơn
    # trọng số cuối (cùng tinh thần SWAD / model soups). Mỗi epoch đánh giá cả hai, giữ bản tốt hơn theo VAL.
    import copy
    ema_decay = float(t.get("ema", 0.0) or 0.0)
    ema_model = copy.deepcopy(model).eval() if ema_decay > 0 else None
    if ema_model is not None:
        for p_ in ema_model.parameters():
            p_.requires_grad_(False)
    fwd = model
    if t.get("compile") and device.type == "cuda":
        try:
            fwd = torch.compile(model)
            log.info("torch.compile bật")
        except Exception as e:
            log.warning(f"torch.compile lỗi ({e}) -> chạy thường")
    penalty = t.get("penalty", "vrex")
    q_dro = torch.ones(max(n_env, 1), device=device) / max(n_env, 1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_train = time.time()

    start_ep, gate_hist = 0, []
    val_sets = build_val_sets(cfg, log)
    if fclip.needed:  # tính trước CLIP toàn ảnh cho mọi ảnh train + VAL rồi giải phóng encoder (tránh tràn VRAM 4 GB)
        fclip.prefill(rows + [r for _, rs in val_sets.values() for r in rs], log, num_workers=min(4, t["num_workers"]))
    best_score, best_ep, val_hist = -1.0, -1, []
    ck = run_dir / "last.pt"
    if ck.exists():
        s = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        scaler.load_state_dict(s["scaler"]); start_ep, gate_hist = s["epoch"] + 1, s.get("gate_hist", [])
        if "q_dro" in s:
            q_dro = s["q_dro"].to(device)
        if ema_model is not None and "ema" in s:
            ema_model.load_state_dict(s["ema"])
        best_score, best_ep, val_hist = s.get("best_score", -1.0), s.get("best_ep", -1), s.get("val_hist", [])
        log.info(f"Tiếp tục từ epoch {start_ep}")
    dump(cfg, run_dir / "config.json")

    for ep in range(start_ep, t["epochs"]):
        model.train()
        fclip.release()  # không giữ encoder CLIP trong VRAM khi train
        beta = 0.0 if ep < t["beta_warmup_epochs"] else t["beta"]
        t0, run_loss, env_loss_sum, env_cnt = time.time(), 0.0, np.zeros(max(n_env, 1)), np.zeros(max(n_env, 1))
        for it, batch in enumerate(dl):
            if full_img:  # ảnh nguyên kích thước (danh sách), không cắt
                xs, g224, y, env, idx = batch
                y, env = y.to(device), env.to(device)
                emb = fclip(g224.to(device), [rows[i]["path"] for i in idx.tolist()])
                with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                    logit = model.forward_images([x_.to(device, non_blocking=True) for x_ in xs], emb, train=True).float()
            else:
                if mil:  # x: mọi mảnh của các ảnh trong batch; counts: số mảnh mỗi ảnh; centers: vùng giữa (cho CLIP)
                    x, counts, centers, y, env, idx = batch
                    counts = counts.to(device)
                else:
                    x, y, env, idx = batch
                    counts = centers = None
                x, y, env = x.to(device, non_blocking=True), y.to(device), env.to(device)
                centers = x if centers is None else centers.to(device, non_blocking=True)
                emb = clip(centers, [rows[i]["path"] for i in idx.tolist()])
                if mil and emb is not None:  # CLIP dùng chung cho mọi mảnh của ảnh đó
                    emb = emb.repeat_interleave(counts, dim=0)
                if t.get("channels_last") and device.type == "cuda":
                    x = x.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                    logit = (fwd(x, emb, counts) if mil else fwd(x, emb)).float()
            present = [e for e in range(n_env) if (env == e).any()]
            if penalty == "irm" and beta > 0:
                scale = torch.ones((), device=device, requires_grad=True)
                losses = torch.stack([F.binary_cross_entropy_with_logits(logit[env == e] * scale, y[env == e])
                                      for e in present])
            else:
                losses = torch.stack([F.binary_cross_entropy_with_logits(logit[env == e], y[env == e])
                                      for e in present])
            if penalty == "groupdro" and len(present) > 1:
                # GroupDRO (Sagawa et al. 2020): tăng trọng số môi trường đang có loss cao
                with torch.no_grad():
                    for k, e in enumerate(present):
                        q_dro[e] *= torch.exp(t.get("dro_eta", 0.01) * losses[k].detach())
                    q_dro /= q_dro.sum()
                w = q_dro[present] / q_dro[present].sum()
                loss = (w * losses).sum()
            else:
                loss = losses.mean()
            if beta > 0 and len(losses) > 1:
                if penalty == "vrex":
                    loss = loss + beta * losses.var(unbiased=False)
                elif penalty == "irm":  # IRMv1 (Arjovsky et al. 2019)
                    pen = torch.stack([torch.autograd.grad(l, scale, create_graph=True)[0] ** 2 for l in losses])
                    loss = loss + beta * pen.mean()
            if use_gates and t["lam"] > 0:
                loss = loss + t["lam"] * torch.sigmoid(model.gate).sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            if ema_model is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema_model.parameters(), model.parameters()):
                        pe.lerp_(pm.detach(), 1.0 - ema_decay)
                    for be, bm in zip(ema_model.buffers(), model.buffers()):
                        be.copy_(bm)
            run_loss += loss.item()
            for k, e in enumerate(present):
                env_loss_sum[e] += losses[k].item(); env_cnt[e] += 1
            if it % 25 == 0 or it == steps - 1:  # ~mỗi 1600 ảnh: kèm tốc độ và thời gian còn lại của epoch
                el = time.time() - t0
                eta = el / (it + 1) * (steps - it - 1) / 60 if it else float("nan")
                log.info(f"ep {ep + 1}/{t['epochs']} it {it + 1}/{steps} loss {loss.item():.4f} beta {beta}"
                         + (f" | {(it + 1) * bsz / el:.0f} ảnh/s | epoch còn ~{eta:.0f} phút" if it else ""))
        fclip.save()
        gates = model.gate_dict()
        gate_hist.append(gates)
        env_mean = {env_names[e]: float(env_loss_sum[e] / max(env_cnt[e], 1)) for e in range(n_env)}
        log.info(f"Epoch {ep}: loss {run_loss / max(1, it + 1):.4f} | loss theo môi trường {env_mean} | "
                 f"{time.time() - t0:.0f}s")
        if gates:
            log.info("Cổng: " + ", ".join(f"{k}={v:.2f}" for k, v in gates.items()))
        if val_sets:
            vres = validate(model, cfg, val_sets)
            sc, key = val_score(vres, t)
            src_best, sd_best = "raw", model.state_dict()
            if ema_model is not None:
                vres_e = validate(ema_model, cfg, val_sets)
                sc_e, _ = val_score(vres_e, t)
                log.info(f"VAL epoch {ep} (EMA): {vres_e}")
                if not np.isnan(sc_e) and (np.isnan(sc) or sc_e > sc):
                    sc, vres, src_best, sd_best = sc_e, vres_e, "ema", ema_model.state_dict()
            val_hist.append(dict(epoch=ep, weights=src_best, **vres))
            log.info(f"VAL epoch {ep}: {vres} | tiêu chí {key}={sc:.4f} ({src_best})")
            if not np.isnan(sc) and sc > best_score:
                best_score, best_ep = sc, ep
                torch.save(sd_best, run_dir / "best.pt")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), epoch=ep, gate_hist=gate_hist, q_dro=q_dro.cpu(),
                        best_score=best_score, best_ep=best_ep, val_hist=val_hist,
                        **({"ema": ema_model.state_dict()} if ema_model is not None else {})), ck)

    train_time = time.time() - t_train
    if (run_dir / "best.pt").exists() and best_ep >= 0:
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
        log.info(f"Dùng checkpoint tốt nhất theo VAL: epoch {best_ep} (điểm {best_score:.4f})")
    elif ema_model is not None and not val_sets:
        model.load_state_dict(ema_model.state_dict()); log.info("Không có VAL: dùng trọng số EMA")
    # Ngưỡng quyết định chọn trên VAL cùng miền (Youden): dùng cho Acc trên bộ sinh lạ thay vì 0.5 cố định.
    # Hợp lệ vì chỉ nhìn dữ liệu tách từ train (DomainBed), không nhìn test.
    val_threshold = None
    if "indomain" in val_sets:
        from .skfallback import roc_curve
        from .evaluate import predict
        name_v, rows_v = val_sets["indomain"]
        y_v = np.array([r["label"] for r in rows_v])
        if len(set(y_v.tolist())) == 2:
            p_v = predict(model, cfg, rows_v, clip_lists=[name_v])
            fpr, tpr, thr = roc_curve(y_v, p_v)
            val_threshold = float(np.clip(thr[int(np.argmax(tpr - fpr))], 1e-4, 1 - 1e-4))
            log.info(f"Ngưỡng chọn trên VAL: {val_threshold:.3f}")
    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    sd = model.state_dict()
    frozen = {n for n, p_ in model.named_parameters() if not p_.requires_grad}
    partial = bool(frozen)
    if partial:  # bỏ trọng số đóng băng (VD CLIP ViT-L ~1,2 GB): nạp lại từ mô hình học sẵn khi đánh giá
        sd = {k: v for k, v in sd.items() if k not in frozen}
    torch.save(dict(model=sd, spec=[g.__dict__ for g in spec], cfg=cfg, use_gates=use_gates, partial=partial,
                    val_threshold=val_threshold, val_threshold_mode='tiles' if mil else cfg['eval'].get('mode', 'center')), run_dir / "final.pt")
    from .evaluate import evaluate_lists
    test_res = evaluate_lists(model, cfg, t["test_lists"], clip_lists=t["test_lists"], log=log, thr=val_threshold)
    result = dict(exp=exp, commit=git_commit(), envs=env_names, n_train=len(rows), params=n_params,
                  groups=[g.name for g in spec], gates=model.gate_dict(), gate_hist=gate_hist, test=test_res,
                  penalty=penalty if t["beta"] > 0 or penalty == "groupdro" else "erm", precision=prec,
                  train_time_s=round(train_time, 1), peak_mem_gb=round(peak_mem, 2),
                  keep_random=bool(t.get("keep_random")), val_score=best_score, best_epoch=best_ep,
                  val_hist=val_hist, val_holdout_env=t.get("val_holdout_env"), val_threshold=val_threshold,
                  ema=ema_decay, head_lr_mult=mult)
    save_json(result, run_dir / "result.json")
    if not t.get("keep_last", False):  # tiết kiệm ổ: final.pt (trọng số tốt nhất theo VAL) là đủ để đánh giá lại
        for f in ("last.pt", "best.pt"):
            (run_dir / f).unlink(missing_ok=True)
    if t.get("keep_final") is False:  # run tinh chỉnh: chỉ cần result.json
        (run_dir / "final.pt").unlink(missing_ok=True)
    log.info(f"Xong {exp}")
    return result


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[], help="ghi đè config, ví dụ train.epochs=3")
    ap.add_argument("--features", help="ví dụ npr hoặc rgb,wpt,npr,fft,clip")
    ap.add_argument("--data", choices=["raw", "debiased"], default="raw")
    ap.add_argument("--protocol", choices=["genimage_sd14", "progan4"])
    ap.add_argument("--mode", choices=["a", "b"])
    ap.add_argument("--envs", help="all | none | upsample,vae_sd ...")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--penalty", choices=["vrex", "irm", "groupdro"], help="kiểu phạt đa môi trường (mặc định vrex)")
    ap.add_argument("--keep-random", action="store_true", help="đối chứng: chọn ngẫu nhiên cùng số nhóm như --keep-from")
    ap.add_argument("--lam", type=float)
    ap.add_argument("--stage", choices=["select", "final"])
    ap.add_argument("--keep-from")
    ap.add_argument("--trunk", help="cnn | vit | timm:<tên> | rgb:<tên> | clip_ft | vit_rgb | resnet50 | convnext | dinov2")
    ap.add_argument("--pretrained", type=int)
    ap.add_argument("--n-train")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    t = cfg["train"]
    if a.features: cfg["features"] = a.features.split(",")
    cfg["data_variant"] = a.data
    if a.protocol == "progan4": t["train_list"] = "progan4_train"
    if a.protocol == "genimage_sd14": t["train_list"] = "genimage_sd14_train"
    if a.mode: t["mode"] = a.mode
    if a.envs: t["envs"] = a.envs if a.envs in ("all", "none") else a.envs.split(",")
    if a.penalty: t["penalty"] = a.penalty
    if a.keep_random: t["keep_random"] = True
    for k in ("beta", "lam", "stage", "trunk", "epochs"):
        if getattr(a, k) is not None: t[k] = getattr(a, k)
    if a.keep_from: t["keep_from"] = a.keep_from
    if a.pretrained is not None: t["pretrained"] = bool(a.pretrained)
    if a.n_train: t["n_train"] = a.n_train if a.n_train == "full" else int(a.n_train)
    for s in (a.seeds or [t["seed"]]):
        c = deepcopy(cfg); c["train"]["seed"] = s
        train_one(c, a.exp if not a.seeds else f"{a.exp}_s{s}", force=a.force)


if __name__ == "__main__":
    main()
