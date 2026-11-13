"""Ngân hàng mô phỏng dấu vết (mục 6).

Biến ảnh thật thành ảnh giả mô phỏng có CÙNG nội dung nhưng mang dấu vết của bộ giải mã.
Mỗi họ mô phỏng = một môi trường cho bước chọn đặc trưng bất biến.

Lệnh:
  python -m src.simulate --family upsample --n 30000
  python -m src.simulate --family all
Đầu ra: <data_root>/sim/real/*.jpg (ảnh thật đã crop + JPEG, dùng chung), <data_root>/sim/<họ>/*.jpg,
        lists/sim_<họ>.csv (thật nhãn 0 + giả nhãn 1, source = tên họ).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from .config import lists_dir, load_config
from .utils import get_device, get_logger, read_list, write_list

log = get_logger("simulate")


# ---------------------------------------------------------------- họ upsampling
def sim_upsample(x: torch.Tensor) -> torch.Tensor:
    """x: [B,3,H,W] trong [0,1], H,W chẵn. Thu nhỏ x2 rồi phóng lại bằng một phép upsampling ngẫu nhiên."""
    small = F.interpolate(x, scale_factor=0.5, mode="area")
    op = random.choice(["nearest", "bilinear", "bicubic", "tconv", "shuffle"])
    if op in ("nearest", "bilinear", "bicubic"):
        kw = {} if op == "nearest" else {"align_corners": False}
        y = F.interpolate(small, scale_factor=2, mode=op, **kw)
    elif op == "tconv":  # transposed conv với kernel ngẫu nhiên, chuẩn hoá để giữ độ sáng
        k = random.choice([3, 4])
        w = torch.rand(3, 1, k, k, device=x.device)
        w = w / w.sum(dim=(2, 3), keepdim=True) * 4
        y = F.conv_transpose2d(small, w, stride=2, padding=(k - 2) // 2 if k == 4 else 0, groups=3)
        y = F.interpolate(y, size=x.shape[-2:], mode="nearest") if y.shape[-2:] != x.shape[-2:] else y
    else:  # pixel-shuffle với conv ngẫu nhiên khởi tạo gần đồng nhất
        w = torch.randn(12, 3, 3, 3, device=x.device) * 0.05
        w[:, :, 1, 1] += torch.eye(3, device=x.device).repeat_interleave(4, dim=0)
        y = F.pixel_shuffle(F.conv2d(small, w, padding=1), 2)
    return y.clamp(0, 1)


# ---------------------------------------------------------------- họ VAE
def load_vae(repo: str, subfolder: str | None, device):
    from diffusers import AutoencoderKL
    from .utils import fp16_unsafe
    dtype = torch.float16 if (device.type == "cuda" and not fp16_unsafe()) else torch.float32
    vae = AutoencoderKL.from_pretrained(repo, subfolder=subfolder, torch_dtype=dtype).to(device).eval()
    return vae, dtype


@torch.no_grad()
def sim_vae(x: torch.Tensor, vae, dtype) -> torch.Tensor:
    z = vae.encode((x * 2 - 1).to(dtype)).latent_dist.mode()  # scale/shift triệt tiêu khi decode ngay
    y = (vae.decode(z).sample.float() + 1) / 2
    bad = ~torch.isfinite(y).flatten(1).all(1)
    if bad.any():  # FP16 tràn số -> làm lại riêng các ảnh lỗi bằng FP32
        vae32 = vae.float()
        z32 = vae32.encode((x[bad] * 2 - 1).float()).latent_dist.mode()
        y[bad] = (vae32.decode(z32).sample.float() + 1) / 2
        vae.to(dtype)
    return y.clamp(0, 1)


def _bad_image(path) -> bool:
    """Ảnh mô phỏng hỏng (đen/đồng màu do NaN, hoặc không mở được) -> cần tạo lại."""
    try:
        a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        black = (a.max(axis=2) <= 2).mean()  # NaN -> 0 khi lưu: vùng đen tuyệt đối
        return a.std() < 1.0 or black > 0.2
    except Exception:
        return True


# ---------------------------------------------------------------- chạy
def _load_crop(path: str, crop: int):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if min(w, h) < crop:
        img = TF.pad(img, [0, 0, max(0, crop - w), max(0, crop - h)], padding_mode="reflect")
    return TF.center_crop(img, [crop, crop])


def run_family(cfg, family: str, n: int | None = None, batch: int = 16, seed: int = 0):
    s = cfg["sim"]
    n = n or s["n_per_family"]
    crop, q = s["crop"], s["jpeg_quality"]
    device = get_device()
    pool = read_list(lists_dir(cfg) / "realpool.csv")
    fam_idx = list(s["vae_repos"].keys()).index(family) + 1 if family != "upsample" else 0
    rng = random.Random(seed * 1000 + fam_idx)
    root = Path(cfg["paths"]["data_root"]) / "sim"
    out_real, out_fake = root / "real", root / family
    # Ưu tiên ảnh mô phỏng ĐÃ CÓ và còn tốt (tiết kiệm thời gian, nhất là khi n nhỏ hơn lần chạy trước),
    # thiếu mới lấy thêm ảnh thật mới theo thứ tự ngẫu nhiên cố định.
    order = rng.sample(pool, len(pool))
    def _key(r):
        return f"{r['source']}_{Path(r['path']).stem}"
    have = [r for r in order if (out_fake / f"{_key(r)}.jpg").exists() and (out_real / f"{_key(r)}.jpg").exists()]
    good = []
    for r in have:
        if len(good) >= n:
            break
        if family == "upsample" or not _bad_image(out_fake / f"{_key(r)}.jpg"):
            good.append(r)
    gset = {id(r) for r in good}
    pool = good + [r for r in order if id(r) not in gset][:max(0, min(n, len(order)) - len(good))]
    log.info(f"[{family}] dùng lại {len(good)} ảnh mô phỏng đã có")
    out_real.mkdir(parents=True, exist_ok=True)
    out_fake.mkdir(parents=True, exist_ok=True)

    rows = []
    todo = []
    for r in pool:
        key = f"{r['source']}_{Path(r['path']).stem}"
        dst_r, dst_f = out_real / f"{key}.jpg", out_fake / f"{key}.jpg"
        rows += [dict(path=str(dst_r), label=0, env=0, source=family, split="sim"),
                 dict(path=str(dst_f), label=1, env=0, source=family, split="sim")]
        if not dst_f.exists() or not dst_r.exists() or id(r) not in gset:
            if dst_f.exists():
                dst_f.unlink(missing_ok=True)  # ảnh VAE lỗi (FP16) -> tạo lại
            todo.append((r["path"], dst_r, dst_f))
    log.info(f"[{family}] {len(pool)} ảnh, cần tạo {len(todo)}")

    vae = dtype = None
    if family != "upsample" and todo:  # ảnh mô phỏng đã có đủ -> chỉ ghi lại danh sách, không cần tải VAE
        repo, sub = s["vae_repos"][family]
        log.info(f"Tải VAE {repo} ({sub})...")
        vae, dtype = load_vae(repo, sub, device)
        if dtype == torch.float32 and device.type == "cuda":
            batch = min(batch, 8)  # FP32 tốn bộ nhớ gấp đôi
            log.info(f"[{family}] GPU GTX 16xx: chạy VAE bằng FP32 (tránh NaN), batch {batch}")

    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        imgs = []
        for src, dst_r, _ in chunk:
            img = _load_crop(src, crop)
            if not dst_r.exists():  # ảnh thật đi qua cùng crop + JPEG -> chỉ khác ở dấu vết
                img.save(dst_r, "JPEG", quality=q)
            imgs.append(TF.to_tensor(img))
        x = torch.stack(imgs).to(device)
        y = sim_upsample(x) if vae is None else sim_vae(x, vae, dtype)
        for (_, _, dst_f), yi in zip(chunk, y.cpu()):
            TF.to_pil_image(yi).save(dst_f, "JPEG", quality=q)
        if (i // batch) % 50 == 0:
            log.info(f"[{family}] {min(i + batch, len(todo))}/{len(todo)}")
    write_list(rows, lists_dir(cfg) / f"sim_{family}.csv")
    log.info(f"[{family}] xong -> lists/sim_{family}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, help="upsample | vae_sd | vae_sdxl | vae_sd3 | vae_flux | all")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    fams = cfg["sim"]["families"] if a.family == "all" else [a.family]
    for f in fams:
        run_family(cfg, f, a.n, a.batch)


if __name__ == "__main__":
    main()
