"""Tách đặc trưng (mục 7): RGB, wavelet (HH và wavelet packet 16 dải), NPR, phổ FFT, CLIP.

Mỗi đặc trưng được chia thành các NHÓM; mỗi nhóm có một cổng ở mô hình (mục 8).
Với crop 256, mọi nhóm không gian có kích thước 64x64.

Lệnh:
  python -m src.features cache-clip --lists genimage_sd14_train genimage_all_val sim_upsample ...
  python -m src.features cache-clip --lists all          # mọi file trong lists/
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import lists_dir, load_config
from .utils import get_device, get_logger, read_list

log = get_logger("features")

BAND_NAMES = ["LL", "LH", "HL", "HH"]
CLIP_DIMS = {"ViT-L-14": 768, "ViT-B-16": 512, "ViT-B-32": 512, "ViT-H-14": 1024}


# ---------------------------------------------------------------- phép biến đổi
def haar_dwt(x: torch.Tensor):
    """[B,C,H,W] -> [LL, LH, HL, HH], mỗi cái [B,C,H/2,W/2] (Haar trực chuẩn)."""
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return [(a + b + c + d) / 2, (a - b + c - d) / 2, (a + b - c - d) / 2, (a - b - c + d) / 2]


def wpt2(x: torch.Tensor):
    """Wavelet packet 2 tầng: 16 dải, thứ tự [LL-LL, LL-LH, ..., HH-HH], mỗi dải [B,3,H/4,W/4]."""
    return [s2 for s1 in haar_dwt(x) for s2 in haar_dwt(s1)]


def npr(x: torch.Tensor):
    """Neighboring Pixel Relationships (Tan et al., CVPR 2024): x - up(down(x)) bằng nearest."""
    up = F.interpolate(F.interpolate(x, scale_factor=0.5, mode="nearest"), scale_factor=2, mode="nearest")
    return x - up


def fft_radial(x: torch.Tensor, n_bins: int = 128, window: bool = False):
    """Phổ biên độ log trung bình theo vòng tròn: [B,n_bins]. Ảnh kích thước bất kỳ (bán kính chuẩn hoá về n_bins).
    window=True: nhân cửa sổ Hann 2D trước FFT để giảm vạch giả do mép ảnh (FFT coi ảnh lặp tuần hoàn)."""
    g = x.mean(1)
    if window:
        hy = torch.hann_window(g.shape[-2], periodic=False, device=g.device, dtype=g.dtype)
        hx = torch.hann_window(g.shape[-1], periodic=False, device=g.device, dtype=g.dtype)
        g = (g - g.mean((-2, -1), keepdim=True)) * (hy[:, None] * hx[None, :])
    mag = torch.log1p(torch.fft.fftshift(torch.fft.fft2(g), dim=(-2, -1)).abs())
    H, W = g.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device) - H // 2,
                            torch.arange(W, device=x.device) - W // 2, indexing="ij")
    r = (torch.sqrt((yy ** 2 + xx ** 2).float()) / (min(H, W) / 2) * (n_bins - 1)).long().clamp(max=n_bins - 1).flatten()
    s = torch.zeros(g.shape[0], n_bins, device=x.device).index_add_(1, r, mag.flatten(1))
    return s / torch.bincount(r, minlength=n_bins).clamp(min=1)


# ---------------------------------------------------------------- nhóm đặc trưng
@dataclass
class Group:
    name: str
    kind: str  # "sp" (không gian) | "vec" (vector)
    dim: int   # số kênh (sp) hoặc số chiều (vec)


def group_spec(features: list[str], clip_arch: str = "ViT-L-14") -> list[Group]:
    sp, vec = [], []
    if "rgb" in features:
        sp.append(Group("rgb", "sp", 48))
    if "hh" in features:
        sp.append(Group("hh1", "sp", 12))
    if "wpt" in features:
        sp += [Group(f"wpt_{a}-{b}", "sp", 3) for a in BAND_NAMES for b in BAND_NAMES]
    if "npr" in features:
        sp.append(Group("npr", "sp", 48))
    if "fft" in features:
        vec += [Group(f"fft_b{i}", "vec", 16) for i in range(8)]
    if "clip" in features:
        vec.append(Group("clip", "vec", CLIP_DIMS.get(clip_arch, 768)))
    return sp + vec


def compute_groups(x: torch.Tensor, spec: list[Group], clip_emb: torch.Tensor | None = None):
    """x: [B,3,256,256] trong [0,1]. Trả về (list tensor không gian [B,c,64,64], list vector [B,d])."""
    names = {g.name for g in spec}
    sp, vec = [], []
    cache = {}
    for g in spec:
        if g.name == "rgb":
            sp.append(F.pixel_unshuffle(x, 4))
        elif g.name == "hh1":
            sp.append(F.pixel_unshuffle(haar_dwt(x)[3], 2))
        elif g.name.startswith("wpt_"):
            if "wpt" not in cache:
                cache["wpt"] = wpt2(x)
            a, b = g.name[4:].split("-")
            sp.append(cache["wpt"][BAND_NAMES.index(a) * 4 + BAND_NAMES.index(b)])
        elif g.name == "npr":
            sp.append(F.pixel_unshuffle(npr(x), 4))
        elif g.name.startswith("fft_b"):
            if "fft" not in cache:
                cache["fft"] = fft_radial(x).split(16, dim=1)
            vec.append(cache["fft"][int(g.name[5:])])
        elif g.name == "clip":
            if clip_emb is None:
                raise ValueError("Cần embedding CLIP (cache hoặc ClipEncoder)")
            vec.append(clip_emb.float())
    assert len(sp) + len(vec) == len(names)
    return sp, vec


# ---------------------------------------------------------------- CLIP
class ClipEncoder(torch.nn.Module):
    """CLIP đóng băng; nhận tensor [B,3,H,W] trong [0,1], tự resize 224 + chuẩn hoá."""
    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, arch="ViT-L-14", pretrained="openai", device=None):
        super().__init__()
        import open_clip
        self.device = device or get_device()
        self.model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()
        from .utils import fp16_unsafe
        self.half = self.device.type == "cuda" and not fp16_unsafe()  # GTX 16xx: FP32 (FP16 chậm/NaN)
        if self.half:
            self.model = self.model.half()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1))
        self.to(self.device)

    @torch.no_grad()
    def forward(self, x):
        x = F.interpolate(x.to(self.device).float(), size=224, mode="bicubic", align_corners=False).clamp(0, 1)
        x = (x - self.mean) / self.std
        if self.half:
            x = x.half()
        e = self.model.encode_image(x).float()
        return e / e.norm(dim=-1, keepdim=True)


def clip_cache_path(cfg, list_name: str) -> Path:
    return Path(cfg["paths"]["cache"]) / f"clip_{cfg['clip']['arch']}_{list_name}.npz"


def _norm(p) -> str:
    """Khoá so khớp đường dẫn ảnh giữa các lần chạy: bỏ phần gốc trước 'aigi-data/' (dữ liệu có thể đã bị dời
    từ D:/aigi-data sang C:/.../aigi-data), không phân biệt hoa thường và dấu gạch."""
    s = str(p).replace("\\", "/").lower()
    k = s.rfind("aigi-data/")
    return s[k + len("aigi-data/"):] if k >= 0 else s


def _known_clip(cfg, skip: Path | None = None) -> dict:
    """Mọi embedding CLIP đã có trong thư mục cache (mọi danh sách, kể cả file chép từ máy khác và file .part dở dang),
    tra theo đường dẫn ảnh -> ảnh đã tính rồi thì không tính lại."""
    d = Path(cfg["paths"]["cache"])
    known = {}
    pat = f"clip_{cfg['clip']['arch']}_*.npz"
    for f in sorted(d.glob(pat)) + sorted(d.glob(pat.replace(".npz", ".part.npz"))):
        if skip is not None and f == skip:
            continue
        try:
            z = np.load(f, allow_pickle=False)
            ok = np.isfinite(z["emb"].astype(np.float32)).all(1)
            for path, e, g in zip(z["paths"], z["emb"], ok):
                if g:  # embedding NaN (FP16 lỗi trên GTX 16xx) -> tính lại
                    known.setdefault(_norm(path), e)
        except Exception as ex:
            log.warning(f"Bỏ qua {f.name}: {ex}")
    return known


def cache_clip(cfg, list_name: str, batch: int = 64, workers: int = 4):
    from .data import CropCache, ImageListDataset
    out = clip_cache_path(cfg, list_name)
    rows = read_list(lists_dir(cfg) / f"{list_name}.csv")
    want = [_norm(r["path"]) for r in rows]
    if out.exists():
        z = np.load(out, allow_pickle=False)
        good = np.isfinite(z["emb"].astype(np.float32)).all(1)
        have = {_norm(p) for p, g in zip(z["paths"], good) if g}
        if all(w in have for w in want):
            log.info(f"Đã có {out.name}, bỏ qua")
            return
        log.info(f"{out.name} chưa đủ ảnh của danh sách hiện tại -> bổ sung phần thiếu")
    part = out.with_name(out.stem + ".part.npz")
    known = _known_clip(cfg)
    miss = [i for i, w in enumerate(want) if w not in known]
    log.info(f"[{list_name}] {len(rows)} ảnh: dùng lại {len(rows) - len(miss)} embedding đã có, cần tính {len(miss)}")
    if miss:
        crops = CropCache(cfg, ["genimage_sd14_train", "genimage_sd14_train_debiased"])  # đọc crop đã cache: nhanh hơn
        ds = ImageListDataset([rows[i] for i in miss], crop=cfg["train"]["crop"], train=False, cache=crops)
        dl = DataLoader(ds, batch_size=batch, num_workers=workers)
        enc = ClipEncoder(cfg["clip"]["arch"], cfg["clip"]["pretrained"])
        dev = enc.device
        log.info(f"[{list_name}] CLIP {cfg['clip']['arch']} chạy trên {dev}" + ("" if dev.type == "cuda" else
                 " — KHÔNG có GPU: rất chậm. Kiểm tra: python -c \"import torch;print(torch.cuda.is_available())\""))
        new_p, new_e, t0, d0 = [], [], time.time(), 0
        every = max(1, 1000 // batch)  # báo tiến độ ~mỗi 1000 ảnh
        for i, (x, _y, _env, idx) in enumerate(dl):
            e = enc(x).cpu().numpy().astype(np.float16)
            for j, k in enumerate(idx.tolist()):
                new_p.append(rows[miss[k]]["path"]); new_e.append(e[j]); known[want[miss[k]]] = e[j]
            done = len(new_p)
            if i == 0:  # lô đầu gồm cả khởi động -> không tính vào tốc độ
                t0, d0 = time.time(), done
                log.info(f"[{list_name}] {done}/{len(miss)} | đang đo tốc độ...")
                continue
            if i % every == 0 or done == len(miss):
                rate = (done - d0) / max(time.time() - t0, 1e-6)
                eta = (len(miss) - done) / max(rate, 1e-6) / 60
                mem = (f" | VRAM {torch.cuda.memory_allocated() / 1e9:.1f}/{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
                       if dev.type == "cuda" else "")
                log.info(f"[{list_name}] {done}/{len(miss)} | {rate:.1f} ảnh/s | còn ~{eta:.0f} phút{mem}")
                part.parent.mkdir(parents=True, exist_ok=True)  # lưu dở dang: Ctrl+C không mất phần đã tính
                np.savez(part, paths=np.array(new_p), emb=np.stack(new_e))
                if dev.type == "cuda" and rate < 5 and done > 1500:
                    log.warning("Quá chậm so với GPU (thường > 10 ảnh/s). Chạy: python scripts/kiem_tra_gpu.py để tìm nguyên nhân.")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, paths=np.array([r["path"] for r in rows]), emb=np.stack([known[w] for w in want]))
    if part.exists():
        part.unlink()
    log.info(f"Lưu {out}")


class FullClipCache:
    """Embedding CLIP của TOÀN ẢNH (ảnh thu nhỏ về 224x224, chỉ dùng cho nhánh ngữ nghĩa).
    Tính một lần cho mỗi đường dẫn rồi nhớ lại (lưu đĩa: <cache>/clip_full_<arch>.pt) -> các epoch/run sau không tính lại."""

    def __init__(self, cfg, device, needed: bool, persist: bool = True):
        self.needed, self.device, self.cfg = needed, device, cfg
        self.enc, self.dirty = None, False
        self.file = Path(cfg["paths"]["cache"]) / f"clip_full_{cfg['clip']['arch']}.pt" if persist else None
        self.mem = {}
        if needed and self.file is not None and self.file.exists():
            self.mem = torch.load(self.file, map_location="cpu", weights_only=False)

    def __call__(self, g224: torch.Tensor, paths: list[str]):
        if not self.needed:
            return None
        keys = [_norm(p) for p in paths]
        miss = [i for i, k in enumerate(keys) if k not in self.mem]
        if miss:
            if self.enc is None:
                self.enc = ClipEncoder(self.cfg["clip"]["arch"], self.cfg["clip"]["pretrained"], self.device)
            fresh = self.enc(g224[miss]).float().cpu()
            for j, i in enumerate(miss):
                self.mem[keys[i]] = fresh[j]
            self.dirty = True
        return torch.stack([self.mem[k] for k in keys]).to(self.device)

    def save(self):
        if self.dirty and self.file is not None:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.mem, self.file); self.dirty = False

    def release(self):
        """Giải phóng encoder CLIP (ViT-L/14 fp32 ~1,7 GB VRAM). Nếu sau này gặp ảnh thiếu thì nạp lại.
        Quan trọng với GPU 4 GB: encoder nằm cùng thân ViT đang train -> tràn VRAM sang RAM, chậm ~20 lần."""
        self.save()
        if self.enc is not None:
            self.enc = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def prefill(self, rows, log=None, batch_size=16, num_workers=4):
        """Tính TRƯỚC embedding cho mọi ảnh trong rows còn thiếu (ảnh thu nhỏ 224x224, không tăng cường, như lúc test),
        lưu cache rồi giải phóng encoder -> khi train không cần giữ CLIP trong VRAM. Chạy một lần, các run sau dùng lại."""
        if not self.needed:
            return
        miss = [r for r in rows if _norm(r["path"]) not in self.mem]
        if log:
            log.info(f"CLIP toàn ảnh: cache có {len(self.mem)} ảnh, cần tính thêm {len(miss)} ảnh")
        if miss:
            dl = DataLoader(_G224Dataset(miss), batch_size=batch_size, num_workers=num_workers, collate_fn=_collate_g224)
            t0 = time.time()
            with torch.no_grad():
                for k, (g, paths) in enumerate(dl):
                    self(g.to(self.device), paths)
                    if k % 100 == 99:
                        self.save()
                        if log:
                            log.info(f"  CLIP toàn ảnh: {(k + 1) * batch_size}/{len(miss)} ảnh ({time.time() - t0:.0f}s)")
            if log:
                log.info(f"CLIP toàn ảnh: xong {len(miss)} ảnh trong {time.time() - t0:.0f}s, cache {len(self.mem)} ảnh")
        self.release()


class _G224Dataset(torch.utils.data.Dataset):
    """Chỉ đọc ảnh và thu nhỏ về 224x224 cho CLIP (giống ImageListDataset mode='full', train=False)."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image
        import torchvision.transforms.functional as TF
        r = self.rows[i]
        try:
            img = Image.open(r["path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return TF.to_tensor(img.resize((224, 224), Image.BICUBIC)), r["path"]


def _collate_g224(batch):
    return torch.stack([g for g, _ in batch]), [p for _, p in batch]


class ClipStore:
    """Tra embedding CLIP theo đường dẫn ảnh từ các file cache; thiếu thì trả mask False."""

    def __init__(self, cfg, list_names: list[str]):
        self.index, chunks, n = {}, [], 0
        for name in list_names:
            p = clip_cache_path(cfg, name)
            if not p.exists():
                log.warning(f"Chưa có cache CLIP cho {name} -> sẽ tính trực tiếp (chậm). Chạy: python -m src.features cache-clip --lists {name}")
                continue
            z = np.load(p, allow_pickle=False)
            for j, path in enumerate(z["paths"]):
                self.index.setdefault(_norm(path), n + j)
            chunks.append(z["emb"])
            n += len(z["emb"])
        self.emb = torch.from_numpy(np.concatenate(chunks)).float() if chunks else None

    def lookup(self, paths: list[str]):
        idx = [self.index.get(_norm(p), -1) for p in paths]
        mask = torch.tensor([i >= 0 for i in idx])
        if self.emb is None or not mask.any():
            return None, mask
        out = torch.zeros(len(paths), self.emb.shape[1])
        out[mask] = self.emb[[i for i in idx if i >= 0]]
        return out, mask


# ---------------------------------------------------------------- đặc trưng thủ công cho Logistic Regression (E2)
@torch.no_grad()
def handcrafted(x: torch.Tensor, features: list[str]) -> torch.Tensor:
    """Vector đặc trưng thủ công [B,D] từ ảnh [B,3,H,W]."""
    out = []
    if "rgb" in features:
        g = x.mean(1, keepdim=True)
        out += [x.mean((2, 3)), x.std((2, 3)), F.adaptive_avg_pool2d(g, 8).flatten(1)]
    if "fft" in features:
        out.append(fft_radial(x))
    if "hh" in features:
        hh = haar_dwt(x)[3]
        out += [hh.abs().mean((2, 3)), hh.std((2, 3)), fft_radial(hh, 64)]
    if "wpt" in features:
        bands = wpt2(x)
        out += [torch.cat([b.abs().mean((2, 3)) for b in bands], 1), torch.cat([b.std((2, 3)) for b in bands], 1)]
    if "npr" in features:
        n = npr(x)
        out += [n.abs().mean((2, 3)), n.std((2, 3)), fft_radial(n, 64)]
    return torch.cat(out, 1) if out else torch.zeros(x.shape[0], 0, device=x.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache-clip"])
    ap.add_argument("--lists", nargs="+", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    names = [p.stem for p in sorted(lists_dir(cfg).glob("*.csv"))] if a.lists == ["all"] else a.lists
    for n in names:
        if not (lists_dir(cfg) / f"{n}.csv").exists():
            print(f"[cache-clip] bỏ qua {n}: chưa có lists/{n}.csv")
            continue
        cache_clip(cfg, n, a.batch, a.workers)


if __name__ == "__main__":
    main()
