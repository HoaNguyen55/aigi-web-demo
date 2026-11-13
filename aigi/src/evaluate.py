"""Đánh giá (mục 9): AUROC/AP/Acc theo từng bộ sinh, độ bền, kiểm tra thiên lệch, xuất bảng, đo hiệu năng.

Lệnh:
  python -m src.evaluate run --runs N2_final_s0 N2_final_s1 --lists genimage_all_val test_chameleon --robust all
  python -m src.evaluate run --runs "E1_*" --lists genimage_all_val genimage_all_val_debiased
  python -m src.evaluate bias-audit --list genimage_sd14_val [--run E1_npr_raw]      # E0
  python -m src.evaluate tables --out paper/tables
  python -m src.evaluate profile --run N2_final_s0                                   # E11
  python -m src.evaluate compare --run N4_b_final_s0 --run-b E1_npr_raw_s0 --list genimage_all_val   # DeLong
  # Baseline chạy bằng code gốc của tác giả (AIDE, C2P-CLIP, DRCT, B-Free...): xuất danh sách ảnh -> chạy code tác giả
  # -> file CSV (path,score) -> nhập vào để tính CÙNG độ đo trên CÙNG ảnh:
  python -m src.evaluate export-lists --lists genimage_all_val test_chameleon --out ext_lists
  python -m src.evaluate import-scores --name AIDE --csv aide_chameleon.csv --list test_chameleon
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from .skfallback import (LogisticRegression, average_precision_score, roc_auc_score, roc_curve,
                         cross_val_predict)
from torch.utils.data import DataLoader

from .config import lists_dir, load_config
from .data import ImageListDataset
from .features import Group
from .utils import get_device, get_logger, load_json, read_list, save_json

log = get_logger("evaluate")


# ---------------------------------------------------------------- độ đo
def metrics(y: np.ndarray, p: np.ndarray, n_boot: int = 0, val_thr: float | None = None) -> dict:
    """thr: ngưỡng chọn trên VAL của run (nếu có) -> acc_vt = Acc tại ngưỡng đó (hợp lệ, không nhìn test)."""
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) < 2:
        out = dict(n=len(y), acc05=float(((p >= 0.5) == y).mean()))
        if val_thr is not None:
            out["acc_vt"] = float(((p >= val_thr) == y).mean())
        return out
    fpr, tpr, thr = roc_curve(y, p)
    j = np.argmax(tpr - fpr)
    t_best = thr[j]
    out = dict(n=int(len(y)), auroc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
               acc05=float(((p >= 0.5) == y).mean()), acc_best=float(((p >= t_best) == y).mean()),
               real_acc05=float((p[y == 0] < 0.5).mean()), fake_acc05=float((p[y == 1] >= 0.5).mean()))
    if val_thr is not None:
        out["acc_vt"] = float(((p >= val_thr) == y).mean()); out["thr_val"] = float(val_thr)
    if n_boot:
        rng, s = np.random.default_rng(0), []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) == 2:
                s.append(roc_auc_score(y[i], p[i]))
        out["auroc_ci95"] = [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]
    return out


def overlap_group(src: str, overlap: dict | None):
    """'seen' nếu bộ sinh dùng autoencoder có trong ngân hàng mô phỏng, 'unseen' nếu không, None nếu chưa xếp."""
    if not overlap:
        return None
    norm = lambda z: re.sub(r"[^a-z0-9]", "", str(z).lower())
    key = norm(src.split("/")[-1])
    for exact in (True, False):
        for grp in ("seen", "unseen"):
            for tok in overlap.get(grp, []):
                t = norm(tok)
                if (key == t) if exact else (t and key.startswith(t)):
                    return grp
    return None


def per_source(rows, probs, n_boot=0, overlap: dict | None = None, thr: float | None = None) -> dict:
    """Mỗi tập con (source): ảnh giả của nó + ảnh thật của nó (nếu không có thì dùng mọi ảnh thật của danh sách).
    Thêm mean_seen / mean_unseen: trung bình trên bộ sinh đã / chưa có trong ngân hàng mô phỏng."""
    y = np.array([r["label"] for r in rows]); src = np.array([r["source"] for r in rows])
    res = {}
    all_real = y == 0
    for s in sorted(set(src)):
        m = src == s
        if not (y[m] == 1).any():
            continue
        if not (y[m] == 0).any():
            m = m | all_real
        res[s] = metrics(y[m], probs[m], n_boot, thr)
    keys = ["auroc", "ap", "acc05", "acc_best"] + (["acc_vt"] if thr is not None else [])
    vals = [v for v in res.values() if "auroc" in v]
    if vals:
        res["mean"] = {k: float(np.mean([v[k] for v in vals])) for k in keys}
    if overlap:
        grp = {s: overlap_group(s, overlap) for s in res if s != "mean"}
        res["overlap"] = grp
        for g in ("seen", "unseen"):
            gv = [res[s] for s, gg in grp.items() if gg == g and "auroc" in res[s]]
            if gv:
                res[f"mean_{g}"] = {k: float(np.mean([v[k] for v in gv])) for k in keys} | {"n_sources": len(gv)}
    res["all"] = metrics(y, probs, n_boot, thr)
    return res


# ---------------------------------------------------------------- biến đổi kiểm tra độ bền
def _jpeg(img, q):
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=int(q))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


class Perturb:
    """Biến đổi PIL dùng khi test (lớp thay cho lambda để chạy được với DataLoader nhiều worker trên Windows/macOS)."""

    def __init__(self, kind: str, level):
        if kind not in ("jpeg", "resize", "blur", "noise", "social", "fakejpeg"):
            raise ValueError(kind)
        self.kind, self.level = kind, level

    def __call__(self, img, r):
        k, lv = self.kind, self.level
        if k == "jpeg":
            return _jpeg(img, lv)
        if k == "resize":
            return img.resize((max(1, int(img.width * lv)), max(1, int(img.height * lv))), Image.BICUBIC)
        if k == "blur":
            return img.filter(ImageFilter.GaussianBlur(lv))
        if k == "noise":
            a = np.asarray(img).astype(np.float32) / 255 + np.random.default_rng(0).normal(0, lv, (img.height, img.width, 3))
            return Image.fromarray((a.clip(0, 1) * 255).astype(np.uint8))
        if k == "social":
            s = 1080 / max(img.size)
            if s < 1:
                img = img.resize((int(img.width * s), int(img.height * s)), Image.BICUBIC)
            return _jpeg(img, 75)
        # fakejpeg (E0): chỉ nén ảnh giả về JPEG -> mô hình dựa vào định dạng sẽ sụt mạnh
        return _jpeg(img, lv) if r["label"] == 1 else img


def make_perturb(kind: str, level):
    return Perturb(kind, level)


def perturb_list(cfg) -> list[tuple[str, object]]:
    out = []
    for kind, levels in cfg["eval"]["robust"].items():
        for lv in levels:
            out.append((f"{kind}{lv}" if kind != "social" else "social", (kind, lv)))
    return out


# ---------------------------------------------------------------- suy luận
@torch.no_grad()
def predict(model, cfg, rows, clip_lists=(), pre_transform=None):
    if torch.cuda.is_available():  # trả lại VRAM cache còn giữ từ lúc train (một phần đã bị Windows đẩy sang RAM):
        import gc; gc.collect(); torch.cuda.empty_cache()  # nếu không, eval dùng lại các block đó -> thrash qua PCIe, chậm 5-10 lần
    from .train import ClipProvider
    from .data import collate_tiles
    device = next(model.parameters()).device if any(True for _ in model.parameters()) else get_device()
    ev = cfg["eval"]
    mil = getattr(model, "is_mil", False)
    if getattr(model, "is_full", False):  # mô hình toàn ảnh: không cắt, cửa sổ trượt trên bản đồ đặc trưng
        from .data import collate_full
        from .features import FullClipCache
        ds = ImageListDataset(rows, crop=ev["crop"], train=False, pre_transform=pre_transform, mode="full")
        dl = DataLoader(ds, batch_size=ev.get("tile_images_per_batch", 4), collate_fn=collate_full,
                        num_workers=cfg["train"]["num_workers"])
        fclip = FullClipCache(cfg, device, model.needs_clip, persist=pre_transform is None)
        model.eval()
        out = []
        for xs, g224, y, env, idx in dl:
            emb = fclip(g224.to(device), [rows[i]["path"] for i in idx.tolist()])
            z = model.forward_images([x_.to(device) for x_ in xs], emb, train=False,
                                     max_windows=int(ev.get("max_tiles", 0) or 0), chunk=ev["batch_size"])
            out.append(torch.sigmoid(z.float()).cpu().numpy())
        fclip.save()
        return np.concatenate(out)
    mode = "tiles" if mil else ev.get("mode", "center")  # mô hình MIL luôn chấm toàn ảnh
    ds = ImageListDataset(rows, crop=ev["crop"], train=False, pre_transform=pre_transform,
                          mode=mode, max_tiles=ev.get("max_tiles", 0))
    # khi ảnh bị biến đổi, embedding CLIP trong cache không còn đúng -> tính lại trực tiếp
    clip = ClipProvider(cfg, list(clip_lists) if pre_transform is None else [], device, getattr(model, "needs_clip", False))
    model.eval()
    out = []
    if mode == "tiles":
        # Chấm điểm TOÀN ẢNH: mọi mảnh crop x crop phủ kín ảnh (không resize) -> gộp điểm các mảnh thành điểm của ảnh.
        # CLIP (đặc trưng ngữ nghĩa toàn cục) tính 1 lần/ảnh trên vùng giữa, giống lúc train, rồi dùng chung cho mọi mảnh.
        agg, bs = ev.get("tile_agg", "mean_logit"), ev["batch_size"]
        dl = DataLoader(ds, batch_size=ev.get("tile_images_per_batch", 4), collate_fn=collate_tiles,
                        num_workers=cfg["train"]["num_workers"])
        for tiles, counts, centers, y, env, idx in dl:
            emb = clip(centers.to(device), [rows[i]["path"] for i in idx.tolist()])
            emb_t = emb.repeat_interleave(counts.to(emb.device), dim=0) if emb is not None else None
            if mil:  # MIL: vector từng mảnh -> gộp có chú ý trong từng ảnh -> 1 logit/ảnh
                h = torch.cat([model.tile_embed(tiles[s:s + bs].to(device), emb_t[s:s + bs] if emb_t is not None else None)
                               .float() for s in range(0, len(tiles), bs)])
                out.append(torch.sigmoid(model.pool(h, counts.to(device))[0]).cpu().numpy())
                continue
            logits = torch.cat([model(tiles[s:s + bs].to(device), emb_t[s:s + bs] if emb_t is not None else None)
                                .float().reshape(-1).cpu() for s in range(0, len(tiles), bs)])
            for z in torch.split(logits, counts.tolist()):
                if agg == "max":
                    out.append(torch.sigmoid(z).max().numpy().reshape(1))
                elif agg == "mean_prob":
                    out.append(torch.sigmoid(z).mean().numpy().reshape(1))
                else:  # mean_logit
                    out.append(torch.sigmoid(z.mean()).numpy().reshape(1))
        return np.concatenate(out)
    dl = DataLoader(ds, batch_size=ev["batch_size"], num_workers=cfg["train"]["num_workers"])
    for x, y, env, idx in dl:
        x = x.to(device)
        emb = clip(x, [rows[i]["path"] for i in idx.tolist()])
        out.append(torch.sigmoid(model(x, emb).float()).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict_detail(model, cfg, paths):
    """Cho demo: chấm từng ảnh (kích thước bất kỳ) và trả về chi tiết từng mảnh.
    Mỗi phần tử: path, size (W,H), positions [(y,x)], tile_probs, attn (MIL) hoặc None, prob, n_tiles, mode."""
    import torchvision.transforms.functional as TF
    from .data import tile_positions
    from .train import ClipProvider
    device = next(model.parameters()).device
    ev = cfg["eval"]; crop, bs = ev["crop"], ev["batch_size"]
    mil = getattr(model, "is_mil", False)
    if getattr(model, "is_full", False):
        from .data import pad_full
        from .features import FullClipCache
        fclip = FullClipCache(cfg, device, model.needs_clip, persist=False)
        model.eval()
        res = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            W, H = img.size
            g224 = TF.to_tensor(img.resize((224, 224), Image.BICUBIC))[None].to(device)
            x = TF.to_tensor(pad_full(img, crop)).to(device)
            emb = fclip(g224, [str(p)])
            h, pos = model.embed_image(x, emb[0] if emb is not None else None, train=False,
                                       max_windows=int(ev.get("max_tiles", 0) or 0), chunk=bs)
            logit, a = model.pool(h, torch.tensor([len(h)], device=h.device))
            res.append(dict(path=str(p), size=(W, H), positions=pos, tile_probs=torch.sigmoid(model.tile_logits(h)).cpu().numpy(),
                            attn=a.cpu().numpy(), prob=float(torch.sigmoid(logit)[0]), n_tiles=len(pos), mode="full", mil=True))
        return res
    mode = "tiles" if mil else ev.get("mode", "center")
    clip = ClipProvider(cfg, [], device, getattr(model, "needs_clip", False))
    model.eval()
    res = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        W, H = img.size
        if min(W, H) < crop:
            img = TF.pad(img, [0, 0, max(0, crop - W), max(0, crop - H)], padding_mode="reflect")
        x = TF.to_tensor(img)
        if mode == "tiles":
            pos = tile_positions(x.shape[1], x.shape[2], crop, ev.get("max_tiles", 0))
        else:
            pos = [((x.shape[1] - crop) // 2, (x.shape[2] - crop) // 2)]
        tiles = torch.stack([x[:, a:a + crop, b:b + crop] for a, b in pos])
        emb = clip(TF.center_crop(x, [crop, crop])[None].to(device), [str(p)])
        emb_t = emb.expand(len(tiles), -1) if emb is not None else None
        chunks = range(0, len(tiles), bs)
        if mil:
            h = torch.cat([model.tile_embed(tiles[s:s + bs].to(device), emb_t[s:s + bs] if emb_t is not None else None)
                           .float() for s in chunks])
            logit, a = model.pool(h, torch.tensor([len(h)], device=h.device))
            tile_p, attn, prob = torch.sigmoid(model.tile_logits(h)), a.cpu().numpy(), float(torch.sigmoid(logit)[0])
        else:
            z = torch.cat([model(tiles[s:s + bs].to(device), emb_t[s:s + bs] if emb_t is not None else None)
                           .float().reshape(-1) for s in chunks])
            tile_p, attn = torch.sigmoid(z), None
            agg = ev.get("tile_agg", "mean_logit")
            prob = float(tile_p.max() if agg == "max" else tile_p.mean() if agg == "mean_prob" else torch.sigmoid(z.mean()))
        res.append(dict(path=str(p), size=(W, H), positions=pos, tile_probs=tile_p.cpu().numpy(), attn=attn,
                        prob=prob, n_tiles=len(pos), mode=mode, mil=mil))
    return res


def sample_rows(rows, n_per, seed=0):
    """Lấy tối đa n_per ảnh cho mỗi (nguồn, nhãn), cố định seed -> mọi run dùng đúng cùng tập con."""
    if not n_per:
        return rows
    rng = np.random.default_rng(seed)
    out, groups = [], {}
    for i, r in enumerate(rows):
        groups.setdefault((r["source"], r["label"]), []).append(i)
    for k in sorted(groups):
        idx = groups[k]
        out += idx if len(idx) <= n_per else sorted(rng.choice(idx, n_per, replace=False).tolist())
    return [rows[i] for i in sorted(out)]


def evaluate_lists(model, cfg, list_names, clip_lists=None, log=log, pre_transform=None, n_boot=0,
                   max_per_source=None, save_dir=None, tag="clean", thr=None):
    res = {}
    for name in list_names:
        p = lists_dir(cfg) / f"{name}.csv"
        if not p.exists():
            log.warning(f"Thiếu {p.name}, bỏ qua")
            continue
        rows = sample_rows(read_list(p), max_per_source)
        probs = predict(model, cfg, rows, clip_lists=[name] if clip_lists is not None else [], pre_transform=pre_transform)
        if save_dir is not None:  # lưu dự đoán: dùng lại cho DeLong / phân tích, không phải chạy lại mô hình
            np.savez_compressed(Path(save_dir) / f"preds_{tag}_{name}.npz", path=np.array([r["path"] for r in rows]),
                                prob=probs, label=np.array([r["label"] for r in rows]),
                                source=np.array([r["source"] for r in rows]))
        res[name] = per_source(rows, probs, n_boot, cfg.get("eval", {}).get("sim_overlap"), thr)
        m = res[name].get("mean", res[name]["all"])
        log.info(f"[{name}] AUROC TB {m.get('auroc', float('nan')):.4f} | AP {m.get('ap', float('nan')):.4f} | "
                 f"Acc@0.5 {m.get('acc05', float('nan')):.4f}")
    return res


def load_run(cfg, run: str):
    from .models import build_model
    s = torch.load(Path(cfg["paths"]["runs"]) / run / "final.pt", map_location="cpu", weights_only=False)
    rcfg = s["cfg"]
    if not s.get("partial", False):  # final.pt đã chứa ĐỦ trọng số -> không cần tải trọng số học sẵn (chạy offline được)
        rcfg["train"]["pretrained"] = False
    spec = [Group(**g) for g in s["spec"]]
    model = build_model(rcfg["train"], spec, s["use_gates"], rcfg.get("clip"))
    model.load_state_dict(s["model"], strict=not s.get("partial", False))
    rcfg["_val_threshold"] = s.get("val_threshold")  # ngưỡng chọn trên VAL lúc train (nếu có)
    mode = "tiles" if getattr(model, "is_mil", False) else cfg.get("eval", {}).get("mode", "center")
    if mode != s.get("val_threshold_mode", "center"):  # ngưỡng phải chọn lại trên VAL với đúng cách chấm điểm (python -m src.evaluate retune-thr)
        f = Path(cfg["paths"]["runs"]) / run / f"val_threshold_{mode}.json"
        if f.exists():
            rcfg["_val_threshold"] = load_json(f)["threshold"]
        else:
            log.warning(f"[{run}] chưa có ngưỡng VAL cho chế độ '{mode}' -> tạm dùng ngưỡng của chế độ 1 vùng giữa. "
                        f"Chạy: python -m src.evaluate retune-thr --runs {run}")
    return model.to(get_device()).eval(), rcfg


def _match_runs(cfg, patterns):
    runs = sorted(p.name for p in Path(cfg["paths"]["runs"]).iterdir() if (p / "final.pt").exists())
    return [r for r in runs if any(fnmatch.fnmatch(r, pat) for pat in patterns)]


def cmd_run(cfg, patterns, lists, robust):
    for run in _match_runs(cfg, patterns):
        model, rcfg = load_run(cfg, run)
        rcfg["paths"], rcfg["eval"] = cfg["paths"], cfg["eval"]
        rd = Path(cfg["paths"]["runs"]) / run
        # clean: toàn bộ ảnh. Độ bền: lấy mẫu cố định max_per_source_robust ảnh/nguồn/nhãn, và chạy thêm
        # "clean_sub" trên ĐÚNG tập con đó để so độ sụt giảm công bằng (cùng ảnh, chỉ khác biến đổi).
        n_sub = cfg["eval"].get("max_per_source_robust")
        variants = [("clean", None, None)]
        if robust:
            variants += [("clean_sub", None, n_sub)] + [(tag, pt, n_sub) for tag, pt in perturb_list(cfg)]
        pre = "" if cfg["eval"].get("mode", "center") == "center" else cfg["eval"]["mode"] + "_"
        for tag, pt, nmax in variants:
            tag = pre + tag  # chế độ tiles ghi ra file riêng (eval_tiles_clean.json...), không đè kết quả 1 vùng giữa
            out = rd / f"eval_{tag}.json"
            done = load_json(out) if out.exists() else {}
            todo = [l for l in lists if l not in done]
            if not todo:
                continue
            log.info(f"[{run}] {tag}: {todo}")
            fn = make_perturb(*pt) if pt else None
            done.update(evaluate_lists(model, rcfg, todo, clip_lists=[], pre_transform=fn,
                                       n_boot=cfg["eval"]["bootstrap"], max_per_source=nmax,
                                       save_dir=rd if tag == pre + "clean" else None, tag=tag,
                                       thr=rcfg.get("_val_threshold")))
            save_json(done, out)


def cmd_retune_thr(cfg, patterns):
    """Chọn lại ngưỡng quyết định trên VAL (Youden, chỉ dữ liệu tách từ train) cho chế độ chấm điểm hiện tại
    (eval.mode). Không cần train lại: dùng final.pt sẵn có. Lưu runs/<run>/val_threshold_<mode>.json."""
    from .train import build_val_sets
    mode = cfg["eval"].get("mode", "center")
    for run in _match_runs(cfg, patterns):
        model, rcfg = load_run(cfg, run)
        rcfg["paths"], rcfg["eval"] = cfg["paths"], cfg["eval"]
        vs = build_val_sets(rcfg, log)
        if "indomain" not in vs:
            log.warning(f"[{run}] không có danh sách VAL ({rcfg['train'].get('val_lists')}) -> bỏ qua"); continue
        name_v, rows_v = vs["indomain"]
        y = np.array([r["label"] for r in rows_v])
        p = predict(model, rcfg, rows_v, clip_lists=[name_v])
        fpr, tpr, thr = roc_curve(y, p)
        t = float(np.clip(thr[int(np.argmax(tpr - fpr))], 1e-4, 1 - 1e-4))
        out = Path(cfg["paths"]["runs"]) / run / f"val_threshold_{mode}.json"
        save_json(dict(mode=mode, max_tiles=cfg["eval"].get("max_tiles", 0), tile_agg=cfg["eval"].get("tile_agg"),
                       threshold=t, val_list=name_v, n=int(len(y)), val_auroc=float(roc_auc_score(y, p))), out)
        log.info(f"[{run}] ngưỡng VAL ({mode}) = {t:.4f} -> {out}")


# ---------------------------------------------------------------- E0: kiểm tra thiên lệch
def cmd_bias_audit(cfg, list_name, run=None):
    rows = read_list(lists_dir(cfg) / f"{list_name}.csv")
    X, y = [], []
    for r in rows:
        try:
            im = Image.open(r["path"])
            w, h = im.size
            X.append([w, h, w * h, int(im.format == "JPEG"), Path(r["path"]).stat().st_size / (w * h)])
            y.append(r["label"])
        except Exception:
            pass
    X, y = np.array(X, float), np.array(y)
    names = ["width", "height", "pixels", "is_jpeg", "bytes_per_pixel"]
    res = {}
    for cols, tag in [([0, 1, 2], "chỉ kích thước"), ([3], "chỉ định dạng"), ([0, 1, 2, 3, 4], "kích thước + định dạng + dung lượng")]:
        Xs = (X[:, cols] - X[:, cols].mean(0)) / (X[:, cols].std(0) + 1e-8)
        p = cross_val_predict(LogisticRegression(max_iter=1000), Xs, y, cv=5, method="predict_proba")[:, 1]
        res[tag] = metrics(y, p)
        log.info(f"Phân loại chỉ bằng {tag} ({[names[c] for c in cols]}): AUROC {res[tag].get('auroc', float('nan')):.4f}")
    if run:
        model, rcfg = load_run(cfg, run)
        rcfg["paths"], rcfg["eval"] = cfg["paths"], cfg["eval"]
        clean = evaluate_lists(model, rcfg, [list_name], clip_lists=[])[list_name]
        fj = evaluate_lists(model, rcfg, [list_name], clip_lists=[], pre_transform=make_perturb("fakejpeg", 96))[list_name]
        res["model_clean"], res["model_fake_jpeg96"] = clean, fj
        a, b = clean.get("mean", clean["all"]), fj.get("mean", fj["all"])
        log.info(f"[{run}] AUROC sạch {a.get('auroc'):.4f} -> khi nén ảnh giả JPEG q96: {b.get('auroc'):.4f} "
                 f"(sụt {a.get('auroc') - b.get('auroc'):+.4f})")
    out = Path(cfg["paths"]["runs"]) / f"bias_audit_{list_name}{'_' + run if run else ''}.json"
    save_json(res, out)
    log.info(f"Lưu {out}")


# ---------------------------------------------------------------- xuất bảng
def cmd_tables(cfg, out_dir):
    import pandas as pd
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for rd in sorted(Path(cfg["paths"]["runs"]).iterdir()):
        files = [("train", rd / "result.json")] + [(f.stem[5:], f) for f in rd.glob("eval_*.json")]
        for tag, f in files:
            if not f.exists():
                continue
            d = load_json(f)
            d = d.get("test", d) if tag == "train" else d
            for lst, per in d.items():
                if not isinstance(per, dict):
                    continue
                for src, m in per.items():
                    if isinstance(m, dict) and "auroc" in m:
                        recs.append(dict(exp=rd.name, variant="clean" if tag == "train" else tag, list=lst, source=src,
                                         **{k: m[k] for k in ("auroc", "ap", "acc05", "acc_best", "acc_vt") if k in m}))
    if not recs:
        log.warning("Chưa có kết quả nào"); return
    df = pd.DataFrame(recs).drop_duplicates(["exp", "variant", "list", "source"], keep="last")
    df.to_csv(out_dir / "all_results.csv", index=False)
    # gộp seed: E_x_s0, E_x_s1 -> E_x (trung bình ± độ lệch chuẩn)
    df["exp_base"] = df["exp"].str.replace(r"_s\d+$", "", regex=True)
    clean = df[df.variant == "clean"]
    for lst, part in clean.groupby("list"):
        piv = part.pivot_table(index="exp_base", columns="source", values="auroc", aggfunc="mean")
        sd = part[part.source == "mean"].groupby("exp_base")["auroc"].std()
        piv["mean_std"] = sd
        piv.to_csv(out_dir / f"auroc_{lst}.csv")
        (out_dir / f"auroc_{lst}.tex").write_text((piv * 100).round(1).to_latex(na_rep="-"), encoding="utf-8")
    # Bảng tổng quát hoá: trung bình mọi bộ sinh / bộ sinh ĐÃ có trong mô phỏng / CHƯA có (kết luận chính dựa vào unseen)
    gen = clean[clean.source.isin(["mean", "mean_seen", "mean_unseen"])]
    if not gen.empty:
        g = gen.groupby(["exp_base", "list", "source"])["auroc"].agg(["mean", "std", "count"]).reset_index()
        g.to_csv(out_dir / "generalization_seen_unseen.csv", index=False)
        piv = gen.pivot_table(index="exp_base", columns=["list", "source"], values="auroc", aggfunc="mean")
        (out_dir / "generalization_seen_unseen.tex").write_text((piv * 100).round(1).to_latex(na_rep="-"), encoding="utf-8")
    # Bảng hiệu năng (độ chính xác đặt cạnh chi phí)
    prof = []
    for rd in sorted(Path(cfg["paths"]["runs"]).iterdir()):
        if (rd / "profile.json").exists():
            prof.append(dict(exp=rd.name, **load_json(rd / "profile.json")))
    if prof:
        pd.DataFrame(prof).to_csv(out_dir / "efficiency.csv", index=False)
    rob = df[(df.source == "mean")]
    if not rob.empty:
        rob.pivot_table(index=["exp_base", "list"], columns="variant", values="auroc").to_csv(out_dir / "robustness_auroc.csv")
    log.info(f"Xuất bảng vào {out_dir}")


# ---------------------------------------------------------------- kiểm định DeLong (so 2 run trên cùng danh sách)
def _delong_cov(y, preds):
    """DeLong nhanh (Sun & Xu 2014) dùng hạng: O(N log N) bộ nhớ O(N) — chạy được với hàng trăm nghìn ảnh."""
    from .skfallback import rankdata
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.stack([rankdata(p) for p in pos]); ty = np.stack([rankdata(q) for q in neg])
    tz = np.stack([rankdata(np.concatenate([p, q])) for p, q in zip(pos, neg)])
    auc = tz[:, :m].sum(1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    return auc, np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n


def _run_preds(cfg, run, list_name, rows):
    """Lấy dự đoán đã lưu (preds_clean_<list>.npz) nếu có; không thì chạy mô hình. Căn theo đường dẫn ảnh."""
    f = Path(cfg["paths"]["runs"]) / run / f"preds_clean_{list_name}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=False)
        m = dict(zip(d["path"].tolist(), d["prob"].tolist()))
        if all(r["path"] in m for r in rows):
            return np.array([m[r["path"]] for r in rows])
    if not (Path(cfg["paths"]["runs"]) / run / "final.pt").exists():
        raise SystemExit(f"{run}: không có final.pt và chưa có preds_clean_{list_name}.npz đủ ảnh")
    model, rcfg = load_run(cfg, run)
    rcfg["paths"], rcfg["eval"] = cfg["paths"], cfg["eval"]
    return predict(model, rcfg, rows, clip_lists=[list_name])


def cmd_compare(cfg, run_a, run_b, list_name):
    """DeLong: AUROC của run A và B trên cùng danh sách có khác nhau có ý nghĩa thống kê không."""
    from .skfallback import norm_cdf
    rows = read_list(lists_dir(cfg) / f"{list_name}.csv")
    probs = [_run_preds(cfg, r, list_name, rows) for r in (run_a, run_b)]
    y = np.array([r["label"] for r in rows])
    auc, cov = _delong_cov(y, np.stack(probs))
    z = (auc[0] - auc[1]) / np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 1e-12))
    p = 2 * (1 - norm_cdf(abs(z)))
    res = dict(list=list_name, run_a=run_a, run_b=run_b, auroc_a=float(auc[0]), auroc_b=float(auc[1]), z=float(z), p_value=float(p))
    save_json(res, Path(cfg["paths"]["runs"]) / f"delong_{run_a}_vs_{run_b}_{list_name}.json")
    log.info(f"{run_a} AUROC {auc[0]:.4f} vs {run_b} {auc[1]:.4f} | z={z:.2f} p={p:.4g} "
             f"({'khác biệt có ý nghĩa' if p < 0.05 else 'chưa đủ ý nghĩa'} ở mức 0,05)")


# ---------------------------------------------------------------- baseline ngoài (code gốc của tác giả)
def cmd_export_lists(cfg, lists, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for n in lists:
        p = lists_dir(cfg) / f"{n}.csv"
        if p.exists():
            rows = read_list(p)
            (out_dir / f"{n}.txt").write_text("\n".join(r["path"] for r in rows), encoding="utf-8")
            log.info(f"{n}: {len(rows)} đường dẫn -> {out_dir / (n + '.txt')}")
    log.info("Chạy code gốc của baseline trên các danh sách này, ghi CSV 2 cột path,score (score = xác suất ẢNH GIẢ), "
             "rồi dùng import-scores.")


def cmd_import_scores(cfg, name, csv_path, list_name):
    """Nhập điểm do code gốc của baseline tạo ra -> tính CÙNG độ đo, CÙNG ảnh, CÙNG cách chia nguồn như mọi run."""
    import csv
    rows = read_list(lists_dir(cfg) / f"{list_name}.csv")
    scores = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            scores[r["path"]] = float(r["score"])
    by_tail = {"/".join(Path(k).parts[-2:]): v for k, v in scores.items()}  # dự phòng khi khác ổ đĩa/thư mục gốc
    keep, probs = [], []
    for r in rows:
        v = scores.get(r["path"], by_tail.get("/".join(Path(r["path"]).parts[-2:])))
        if v is not None:
            keep.append(r); probs.append(v)
    cover = len(keep) / max(1, len(rows))
    if cover < 0.99:
        log.warning(f"{name}/{list_name}: chỉ khớp {len(keep)}/{len(rows)} ảnh ({cover:.1%}) — kết quả KHÔNG so được "
                    f"trực tiếp với các run khác nếu thiếu nhiều")
    probs = np.array(probs, dtype=float)
    if probs.min() < 0 or probs.max() > 1:  # điểm dạng logit -> đưa về xác suất
        probs = 1 / (1 + np.exp(-probs))
    rd = Path(cfg["paths"]["runs"]) / f"EXT_{name}"; rd.mkdir(parents=True, exist_ok=True)
    out = rd / "eval_clean.json"
    done = load_json(out) if out.exists() else {}
    done[list_name] = per_source(keep, probs, cfg["eval"]["bootstrap"], cfg["eval"].get("sim_overlap"))
    done[list_name]["coverage"] = cover
    save_json(done, out)
    np.savez_compressed(rd / f"preds_clean_{list_name}.npz", path=np.array([r["path"] for r in keep]), prob=probs,
                        label=np.array([r["label"] for r in keep]), source=np.array([r["source"] for r in keep]))
    save_json(dict(external=True, name=name, source_csv=str(csv_path)), rd / "external.json")
    m = done[list_name].get("mean", done[list_name]["all"])
    log.info(f"EXT_{name} [{list_name}] AUROC TB {m.get('auroc', float('nan')):.4f} | phủ {cover:.1%}")


def cmd_import_dir(cfg, d):
    """Quy ước thư mục: <d>/<TÊN_BASELINE>/<tên danh sách>.csv (cột path,score)."""
    d = Path(d)
    for sub in sorted(p for p in d.iterdir() if p.is_dir()):
        for f in sorted(sub.glob("*.csv")):
            if (lists_dir(cfg) / f"{f.stem}.csv").exists():
                cmd_import_scores(cfg, sub.name, f, f.stem)
            else:
                log.warning(f"{f}: không có lists/{f.stem}.csv tương ứng -> bỏ qua")


# ---------------------------------------------------------------- E11: hiệu năng
@torch.no_grad()
def cmd_profile(cfg, run):
    from torch.utils.flop_counter import FlopCounterMode
    model, rcfg = load_run(cfg, run)
    dev = get_device()
    x = torch.rand(1, 3, rcfg["train"]["crop"], rcfg["train"]["crop"], device=dev)
    clip_dim = next((g.dim for g in getattr(model, "spec", []) if g.name == "clip"), 0)
    emb = torch.zeros(1, clip_dim, device=dev) if clip_dim else None
    with FlopCounterMode(display=False) as fc:
        model(x, emb)
    gflops = fc.get_total_flops() / 1e9
    clip_gflops, clip_params = 0.0, 0.0
    if clip_dim:  # nhóm CLIP cần chạy bộ mã hoá CLIP khi suy luận -> cộng chi phí vào cho công bằng
        try:
            from .features import ClipEncoder
            enc = ClipEncoder(rcfg["clip"]["arch"], rcfg["clip"]["pretrained"], dev)
            with FlopCounterMode(display=False) as fc2:
                enc(x)
            clip_gflops = fc2.get_total_flops() / 1e9
            clip_params = sum(p.numel() for p in enc.parameters()) / 1e6
        except Exception as e:
            log.warning(f"Không đo được CLIP: {e}")
    bs = 32
    xb = x.repeat(bs, 1, 1, 1); eb = emb.repeat(bs, 1) if emb is not None else None
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(3): model(xb, eb)
    if dev.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10): model(xb, eb)
    if dev.type == "cuda": torch.cuda.synchronize()
    dt = (time.time() - t0) / (10 * bs)
    rj = Path(cfg["paths"]["runs"]) / run / "result.json"
    tr = load_json(rj) if rj.exists() else {}
    res = dict(params_M=sum(p.numel() for p in model.parameters()) / 1e6 + clip_params,
               gflops=gflops + clip_gflops, gflops_without_clip=gflops,
               ms_per_image=dt * 1000, images_per_s=1 / dt,
               peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None,
               train_time_h=tr.get("train_time_s", 0) / 3600, train_peak_vram_gb=tr.get("peak_mem_gb"),
               device=str(dev), batch=bs, note="ms/ảnh đo không tính đọc ảnh; CLIP (nếu dùng) đã cộng vào FLOPs/tham số")
    save_json(res, Path(cfg["paths"]["runs"]) / run / "profile.json")
    log.info(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "bias-audit", "tables", "profile", "compare", "export-lists", "import-scores",
                                    "import-dir", "retune-thr"])
    ap.add_argument("--mode", choices=["center", "tiles"], help="center: 1 vùng giữa | tiles: phủ kín toàn ảnh (mặc định theo config)")
    ap.add_argument("--max-tiles", type=int, help="tiles: giới hạn số mảnh mỗi ảnh (0 = phủ kín)")
    ap.add_argument("--dir", default="ext_scores")
    ap.add_argument("--name")
    ap.add_argument("--csv")
    ap.add_argument("--run-b")
    ap.add_argument("--runs", nargs="*", default=["*"])
    ap.add_argument("--run")
    ap.add_argument("--lists", nargs="*", default=["genimage_all_val"])
    ap.add_argument("--list")
    ap.add_argument("--robust", choices=["none", "all"], default="none")
    ap.add_argument("--out", default="paper/tables")
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    if a.mode:
        cfg["eval"]["mode"] = a.mode
    if a.max_tiles is not None:
        cfg["eval"]["max_tiles"] = a.max_tiles
    if a.cmd == "retune-thr":
        cmd_retune_thr(cfg, [a.run] if a.run else a.runs)
    elif a.cmd == "run":
        cmd_run(cfg, a.runs, a.lists, a.robust == "all")
    elif a.cmd == "bias-audit":
        cmd_bias_audit(cfg, a.list, a.run)
    elif a.cmd == "tables":
        cmd_tables(cfg, a.out)
    elif a.cmd == "profile":
        for r in ([a.run] if a.run else _match_runs(cfg, a.runs)):
            cmd_profile(cfg, r)
    elif a.cmd == "export-lists":
        cmd_export_lists(cfg, a.lists, a.out if a.out != "paper/tables" else "ext_lists")
    elif a.cmd == "import-dir":
        cmd_import_dir(cfg, a.dir)
    elif a.cmd == "import-scores":
        cmd_import_scores(cfg, a.name, a.csv, a.list)
    elif a.cmd == "compare":
        cmd_compare(cfg, a.run, a.run_b, a.list)


if __name__ == "__main__":
    main()
