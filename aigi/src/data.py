"""Dữ liệu (mục 5): tạo danh sách ảnh, khử thiên lệch JPEG/kích thước, Dataset crop gốc.

Lệnh:
  python -m src.data make-lists --source genimage
  python -m src.data make-lists --source forensynths
  python -m src.data make-lists --source folder --name chameleon --root D:/aigi-data/raw/test/chameleon
  python -m src.data make-lists --source benchmarks          # mọi mục trong config.test_benchmarks
  python -m src.data make-lists --source realpool
  python -m src.data build-debiased --list genimage_sd14_train
  python -m src.data stats --list genimage_sd14_train
"""
from __future__ import annotations

import argparse
import io
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from .config import load_config, lists_dir
from .utils import get_logger, list_images, read_list, write_list

log = get_logger("data")

REAL_NAMES = {"real", "0_real", "nature", "reals"}
FAKE_NAMES = {"fake", "1_fake", "ai", "fakes", "synthetic"}


# ---------------------------------------------------------------- danh sách
def _rows(paths, label, source, split, env=0):
    return [dict(path=str(p), label=label, env=env, source=source, split=split) for p in paths]


def _cap(paths, n):
    """Giữ tối đa n ảnh (chọn ngẫu nhiên cố định seed 0, rồi sắp lại theo tên). n None/0 = giữ hết.
    Dùng khi trên ổ có nhiều ảnh hơn N_TRAIN (ví dụ đã tải 25k/lớp nhưng muốn train 12k/lớp)."""
    paths = sorted(paths)
    if not n or len(paths) <= int(n):
        return paths
    return sorted(random.Random(0).sample(paths, int(n)))


def cut_list(cfg, name, n, out):
    """Danh sách con: tối đa n ảnh mỗi (nguồn, nhãn), seed 0 cố định -> mọi run dùng đúng cùng ảnh.
    Dùng cho profile vitnhanh (test/môi trường nhỏ hơn để chạy xong trong ngày)."""
    src = lists_dir(cfg) / f"{name}.csv"
    if not src.exists():
        log.warning(f"Chưa có {src.name}, bỏ qua"); return
    rows = read_list(src)
    rng, groups = random.Random(0), {}
    for i, r in enumerate(rows):
        groups.setdefault((r["source"], r["label"]), []).append(i)
    keep = []
    for k in sorted(groups):
        idx = groups[k]
        keep += idx if len(idx) <= n else rng.sample(idx, n)
    sub = [rows[i] for i in sorted(keep)]
    write_list(sub, lists_dir(cfg) / f"{out}.csv")
    log.info(f"{out}: {len(sub)}/{len(rows)} ảnh (tối đa {n} mỗi nguồn x nhãn)")


def lists_genimage(cfg):
    g = cfg["genimage"]
    root = Path(g["root"])
    all_val = []
    for name, sub in g["generators"].items():
        base = root / sub
        for split in ("train", "val"):
            fake = list_images(base / split / g["fake_dir"])
            real = list_images(base / split / g["real_dir"])
            if not fake and not real:
                log.warning(f"[genimage] không thấy ảnh ở {base / split} — kiểm tra lại tên thư mục trong config")
                continue
            if split == "train" and name == g.get("train_generator", "sd14"):
                real, fake = _cap(real, g.get("max_train_per_class")), _cap(fake, g.get("max_train_per_class"))
            rows = _rows(real, 0, name, split) + _rows(fake, 1, name, split)
            if split == "train" and name == g.get("train_generator", "sd14"):
                # Tập VAL thật sự: tách val_frac ảnh từ thư mục TRAIN (phân tầng theo nhãn, seed cố định).
                # Thư mục "val" của GenImage là TẬP TEST chuẩn -> không bao giờ dùng để chỉnh tham số.
                rng = random.Random(0)
                frac = float(g.get("val_frac", 0.05))
                hold = set()
                for lab in (0, 1):
                    idx = [i for i, r in enumerate(rows) if r["label"] == lab]
                    hold |= set(rng.sample(idx, int(len(idx) * frac)))
                tv = [dict(r, split="trainval") for i, r in enumerate(rows) if i in hold]
                rows = [r for i, r in enumerate(rows) if i not in hold]
                write_list(tv, lists_dir(cfg) / f"genimage_{name}_trainval.csv")
                log.info(f"genimage_{name}_trainval (VAL tách từ train): {len(tv)} ảnh")
            write_list(rows, lists_dir(cfg) / f"genimage_{name}_{split}.csv")
            log.info(f"genimage_{name}_{split}: {sum(r['label'] == 0 for r in rows)} thật, "
                     f"{sum(r['label'] == 1 for r in rows)} giả" + ("  (= TEST chuẩn của GenImage)" if split == "val" else ""))
            if split == "val":
                all_val += rows
    clean = g.get("clean_train_dir")  # do src.prepare_data (--plan fit75) tạo sẵn: ảnh đã khử thiên lệch lúc tải
    if clean and (root / clean).exists():
        cd = root / clean
        n_cl = g.get("max_clean_per_class")
        rows = _rows(_cap(list_images(cd / g["real_dir"]), n_cl), 0, g.get("train_generator", "sd14"), "train") + \
            _rows(_cap(list_images(cd / g["fake_dir"]), n_cl), 1, g.get("train_generator", "sd14"), "train")
        rng = random.Random(0)
        hold = set()
        for lab in (0, 1):
            idx = [i for i, r in enumerate(rows) if r["label"] == lab]
            hold |= set(rng.sample(idx, int(len(idx) * float(g.get("val_frac", 0.05)))))
        tn = g.get("train_generator", "sd14")
        write_list([dict(r, split="trainval") for i, r in enumerate(rows) if i in hold],
                   lists_dir(cfg) / f"genimage_{tn}_trainval_debiased.csv")
        write_list([r for i, r in enumerate(rows) if i not in hold], lists_dir(cfg) / f"genimage_{tn}_train_debiased.csv")
        log.info(f"genimage_{tn}_train_debiased (đã khử thiên lệch lúc tải): {len(rows) - len(hold)} ảnh, VAL {len(hold)}")
    if all_val:
        write_list(all_val, lists_dir(cfg) / "genimage_all_val.csv")
        hard = set(g.get("hard4", []))
        # ảnh thật dùng chung (bản arrow không gắn bộ sinh cho ảnh thật) phải đi kèm để tính được AUROC
        write_list([r for r in all_val if r["source"] in hard or (r["label"] == 0 and r["source"] == "real_shared")],
                   lists_dir(cfg) / "genimage_hard4_val.csv")


def lists_forensynths(cfg):
    f = cfg["forensynths"]
    rows = []
    for c in f["classes"]:
        base = Path(f["root"]) / "train" / c
        rows += _rows(list_images(base / "0_real"), 0, "progan", "train")
        rows += _rows(list_images(base / "1_fake"), 1, "progan", "train")
    # VAL tách 5% từ train (cùng cách như GenImage)
    rng = random.Random(0)
    hold = set()
    for lab in (0, 1):
        idx = [i for i, r in enumerate(rows) if r["label"] == lab]
        hold |= set(rng.sample(idx, int(len(idx) * float(cfg["genimage"].get("val_frac", 0.05)))))
    write_list([dict(r, split="trainval") for i, r in enumerate(rows) if i in hold], lists_dir(cfg) / "progan4_trainval.csv")
    rows = [r for i, r in enumerate(rows) if i not in hold]
    write_list(rows, lists_dir(cfg) / "progan4_train.csv")
    log.info(f"progan4_train: {Counter(r['label'] for r in rows)} | progan4_trainval: {len(hold)}")


def _split_real_fake(folder: Path):
    real, fake = [], []
    for d in folder.iterdir():
        if not d.is_dir():
            continue
        if d.name.lower() in REAL_NAMES:
            real += list_images(d)
        elif d.name.lower() in FAKE_NAMES:
            fake += list_images(d)
    return real, fake


def lists_folder(cfg, name: str, root: str):
    """Thư mục benchmark: root/{real,fake} hoặc root/<tập con>/{real,fake} (nhiều cấp cũng được)."""
    root = Path(root)
    rows = []
    real, fake = _split_real_fake(root) if root.exists() else ([], [])
    if real or fake:
        rows = _rows(real, 0, name, "test") + _rows(fake, 1, name, "test")
    else:
        for sub in sorted(p for p in root.rglob("*") if p.is_dir()):
            r, f = _split_real_fake(sub)
            if r or f:
                src = f"{name}/{sub.relative_to(root).as_posix()}"
                rows += _rows(r, 0, src, "test") + _rows(f, 1, src, "test")
    if not rows:
        log.warning(f"[{name}] không tìm thấy thư mục real/fake trong {root}")
        return
    # tập con chỉ có ảnh giả: ghép ảnh thật chung của benchmark (nếu có) khi đánh giá — xem evaluate.py
    write_list(rows, lists_dir(cfg) / f"test_{name}.csv")
    log.info(f"test_{name}: {len(rows)} ảnh, {len({r['source'] for r in rows})} tập con")


def lists_realpool(cfg):
    rows = []
    for src, folder in cfg["real_pool"].items():
        imgs = list_images(folder)
        rows += _rows(imgs, 0, src, "pool")
        log.info(f"realpool/{src}: {len(imgs)} ảnh")
    write_list(rows, lists_dir(cfg) / "realpool.csv")


# ---------------------------------------------------------------- khử thiên lệch
def _jpeg_one(args):
    src, dst, q = args
    dst = Path(dst)
    if dst.exists():
        return str(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src).convert("RGB").save(dst, "JPEG", quality=q)
    return str(dst)


# Bảng lượng tử độ sáng chuẩn IJG (dùng để ước lượng chất lượng JPEG; so theo tổng nên không phụ thuộc thứ tự zigzag)
_IJG_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29,
             51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121,
             120, 101, 72, 92, 95, 98, 112, 100, 103, 99]


def estimate_jpeg_quality(img) -> int | None:
    """Ước lượng chất lượng JPEG (1-100) từ bảng lượng tử; None nếu không phải JPEG."""
    try:
        q = getattr(img, "quantization", None)
        if not q or (img.format or "").upper() != "JPEG":
            return None
        target = sum(q[0])
    except Exception:
        return None
    best, best_d = None, None
    for quality in range(1, 101):
        scale = 5000 / quality if quality < 50 else 200 - 2 * quality
        tsum = sum(min(255, max(1, (v * scale + 50) // 100)) for v in _IJG_LUMA)
        d = abs(tsum - target)
        if best_d is None or d < best_d:
            best, best_d = quality, d
    return best


def grommelt_real_ok(img, d: dict) -> bool:
    """Ảnh thật được giữ theo Grommelt et al. 2024: JPEG chất lượng 96 và CẢ HAI cạnh trong [450, 550]."""
    w, h = img.size
    lo, hi = d.get("real_min_side", 450), d.get("real_max_side", 550)
    if not (lo <= w <= hi and lo <= h <= hi):
        return False
    q = d.get("real_jpeg_quality", 96)
    return q in (None, 0) or estimate_jpeg_quality(img) == q


def _size_ok(args):
    path, d = args
    try:
        return grommelt_real_ok(Image.open(path), d)  # chỉ đọc header + bảng lượng tử
    except Exception:
        return False


def build_debiased(cfg, list_name: str, fakes_only: bool = False):
    """Ảnh giả -> JPEG q96 (như ảnh thật GenImage); ảnh thật -> lọc theo Grommelt et al. 2024 (JPEG q96, cạnh 450-550).
    fakes_only=True (dùng cho TẬP TEST, như DDA): chỉ nén ảnh giả, giữ nguyên mọi ảnh thật."""
    d = cfg["debias"]
    rows = read_list(lists_dir(cfg) / f"{list_name}.csv")
    out_root = Path(cfg["paths"]["data_root"]) / "debiased" / list_name
    fakes = [r for r in rows if r["label"] == 1]
    reals = [r for r in rows if r["label"] == 0]
    jobs = []
    for i, r in enumerate(fakes):
        dst = out_root / "fake" / r["source"] / f"{i:07d}_{Path(r['path']).stem}.jpg"
        jobs.append((r["path"], str(dst), d["jpeg_quality"]))
    with ProcessPoolExecutor(d["workers"]) as ex:
        new_paths = list(ex.map(_jpeg_one, jobs, chunksize=64))
        keep = [True] * len(reals) if fakes_only else \
            list(ex.map(_size_ok, [(r["path"], d) for r in reals], chunksize=64))
    new_rows = [dict(r, path=p) for r, p in zip(fakes, new_paths)]
    new_rows += [r for r, k in zip(reals, keep) if k]
    write_list(new_rows, lists_dir(cfg) / f"{list_name}_debiased.csv")
    log.info(f"{list_name}_debiased: giữ {sum(keep)}/{len(reals)} ảnh thật, nén {len(fakes)} ảnh giả")


def stats(cfg, list_name: str, sample: int = 2000):
    rows = read_list(lists_dir(cfg) / f"{list_name}.csv")
    log.info(f"{list_name}: {len(rows)} ảnh | nhãn {Counter(r['label'] for r in rows)} | nguồn {Counter(r['source'] for r in rows)}")
    random.seed(0)
    for lab in (0, 1):
        sub = random.sample([r for r in rows if r["label"] == lab], min(sample, sum(r["label"] == lab for r in rows)))
        sizes = np.array([Image.open(r["path"]).size for r in sub]) if sub else np.zeros((0, 2))
        fmts = Counter(Path(r["path"]).suffix.lower() for r in sub)
        if len(sizes):
            log.info(f"  nhãn {lab}: cạnh ngắn trung vị {np.median(sizes.min(1)):.0f}, cạnh dài trung vị "
                     f"{np.median(sizes.max(1)):.0f}, định dạng {dict(fmts)}")


# ---------------------------------------------------------------- Dataset
def subsample(rows, n, seed=0):
    """Lấy n ảnh cân bằng thật/giả (n='full' hoặc None: giữ nguyên)."""
    if n in (None, "full") or n >= len(rows):
        return rows
    rng = random.Random(seed)
    out = []
    for lab in (0, 1):
        part = [r for r in rows if r["label"] == lab]
        out += rng.sample(part, min(len(part), n // 2))
    rng.shuffle(out)
    return out


def random_degrade(img: Image.Image, p: float) -> Image.Image:
    """JPEG q~U(30,100) và blur sigma~U(0.1,3), mỗi cái xác suất p (giống AIDE)."""
    if random.random() < p:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 3.0)))
    if random.random() < p:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(30, 100))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img


# ---------------------------------------------------------------- cache crop KHÔNG MẤT DỮ LIỆU (tối ưu tốc độ train)
# Mỗi ảnh được giải nén MỘT lần, lấy vùng giữa size x size (pad phản chiếu nếu nhỏ, KHÔNG resize, KHÔNG nén lại)
# và lưu nguyên điểm ảnh uint8 vào file .npy memmap. Train đọc thẳng -> bỏ bước giải nén JPEG/PNG mỗi epoch.
# Chỉ dùng cho TRAIN. Đánh giá (kể cả độ bền: resize, mạng xã hội) luôn đọc ảnh gốc để đúng chuẩn.
def crop_cache_paths(cfg, list_name: str):
    d = Path(cfg["paths"]["cache"]) / "crops"
    return d / f"{list_name}.npy", d / f"{list_name}.paths.txt"


def _center_region(path: str, size: int) -> np.ndarray:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return np.zeros((size, size, 3), np.uint8)
    w, h = img.size
    if min(w, h) < size:
        img = TF.pad(img, [0, 0, max(0, size - w), max(0, size - h)], padding_mode="reflect")
    w, h = img.size
    l, t = (w - size) // 2, (h - size) // 2
    return np.asarray(img.crop((l, t, l + size, t + size)), dtype=np.uint8)


def build_crop_cache(cfg, list_name: str, workers: int | None = None):
    size = int(cfg.get("cache", {}).get("crop_size", cfg["train"]["crop"]))
    npy, txt = crop_cache_paths(cfg, list_name)
    if npy.exists() and txt.exists():
        log.info(f"Đã có cache crop {list_name} -> bỏ qua"); return
    rows = read_list(lists_dir(cfg) / f"{list_name}.csv")
    npy.parent.mkdir(parents=True, exist_ok=True)
    tmp = npy.with_suffix(".tmp.npy")
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(rows), size, size, 3))
    workers = workers or cfg.get("debias", {}).get("workers", 8)
    with ProcessPoolExecutor(workers) as ex:
        for i, arr in enumerate(ex.map(_center_region, [r["path"] for r in rows], [size] * len(rows), chunksize=64)):
            mm[i] = arr
            if i % 20000 == 0:
                log.info(f"  cache {list_name}: {i}/{len(rows)}")
    mm.flush(); del mm
    tmp.replace(npy)
    txt.write_text("\n".join(r["path"] for r in rows), encoding="utf-8")
    log.info(f"Cache crop {list_name}: {len(rows)} ảnh x {size}px -> {npy} ({npy.stat().st_size / 1e9:.1f} GB)")


def _pkey(p) -> str:
    """Khoá đường dẫn bền với việc dời thư mục dữ liệu (D:/aigi-data -> C:/.../aigi-data)."""
    s = str(p).replace("\\", "/").lower()
    k = s.rfind("aigi-data/")
    return s[k + len("aigi-data/"):] if k >= 0 else s


class CropCache:
    """Tra ảnh đã cache theo đường dẫn. Mở memmap lười (mỗi worker DataLoader tự mở)."""

    def __init__(self, cfg, list_names):
        self.files, self.index = [], {}
        for n in list_names:
            npy, txt = crop_cache_paths(cfg, n)
            if npy.exists() and txt.exists():
                k = len(self.files)
                self.files.append(str(npy))
                for i, p in enumerate(txt.read_text(encoding="utf-8").splitlines()):
                    self.index.setdefault(_pkey(p), (k, i))
        self._mm = {}

    def __len__(self):
        return len(self.index)

    def get(self, path):
        hit = self.index.get(_pkey(path))
        if hit is None:
            return None
        k, i = hit
        if k not in self._mm:
            self._mm[k] = np.load(self.files[k], mmap_mode="r")
        return Image.fromarray(np.array(self._mm[k][i]))

    def __getstate__(self):  # không gửi memmap đang mở sang worker
        s = self.__dict__.copy(); s["_mm"] = {}
        return s


def tile_positions(h: int, w: int, size: int, max_tiles: int = 0) -> list[tuple[int, int]]:
    """Góc trên-trái các mảnh size x size PHỦ KÍN ảnh h x w (mảnh ở mép chồng lấn nhẹ). KHÔNG resize ảnh.
    max_tiles > 0: giới hạn số mảnh (lưới thưa trải đều toàn ảnh) để chạy nhanh với ảnh rất lớn."""
    import math
    ny, nx = max(1, math.ceil(h / size)), max(1, math.ceil(w / size))
    if max_tiles and ny * nx > max_tiles:
        k = math.sqrt(max_tiles / (ny * nx))
        ny, nx = max(1, int(ny * k)), max(1, int(nx * k))
    ys = np.linspace(0, h - size, ny).round().astype(int).tolist() if ny > 1 else [(h - size) // 2]
    xs = np.linspace(0, w - size, nx).round().astype(int).tolist() if nx > 1 else [(w - size) // 2]
    return [(y, x) for y in ys for x in xs]


def pad_full(img: Image.Image, crop: int = 256, mult: int = 4) -> Image.Image:
    """Chế độ toàn ảnh: KHÔNG cắt, KHÔNG thu nhỏ. Chỉ đệm phản chiếu ở mép phải/dưới để mỗi cạnh >= crop
    và là bội của mult (yêu cầu của space-to-depth / Haar 2 mức)."""
    w, h = img.size
    W, H = max(w, crop), max(h, crop)
    W, H = W + (-W) % mult, H + (-H) % mult
    if (W, H) != (w, h):
        img = TF.pad(img, [0, 0, W - w, H - h], padding_mode="reflect")
    return img


def collate_full(batch):
    """Ảnh khác kích thước -> giữ dạng danh sách (không ghép tensor)."""
    xs, g224, y, env, idx = zip(*batch)
    return list(xs), torch.stack(g224), torch.stack(y), torch.stack(env), torch.stack(idx)


def collate_tiles(batch):
    """Gộp batch ở chế độ tiles: mỗi ảnh có số mảnh khác nhau -> nối mảnh lại, kèm số mảnh của từng ảnh."""
    tiles, centers, y, env, idx = zip(*batch)
    return (torch.cat(tiles), torch.tensor([t.shape[0] for t in tiles]), torch.stack(centers),
            torch.stack(y), torch.stack(env), torch.stack(idx))


class ImageListDataset(Dataset):
    """Trả về (ảnh [3,crop,crop] trong [0,1], nhãn, env, chỉ số dòng). Không resize: crop gốc (pad nếu nhỏ)."""

    def __init__(self, rows, crop=256, train=True, aug_prob=0.0, pre_transform=None, cache: CropCache | None = None,
                 mode: str = "center", max_tiles: int = 0, train_tiles: int = 0):
        """mode (chỉ khi train=False): 'center' = 1 vùng giữa crop x crop (cách cũ);
        'tiles' = chia ảnh thành các mảnh crop x crop phủ kín TOÀN ẢNH (không resize), trả thêm vùng giữa cho CLIP."""
        self.rows, self.crop, self.train, self.aug_prob = rows, crop, train, aug_prob
        self.mode, self.max_tiles = mode, int(max_tiles or 0)
        self.train_tiles = int(train_tiles or 0)  # >0 (MIL): mỗi ảnh train lấy tối đa K mảnh ngẫu nhiên trên TOÀN ẢNH
        self.pre_transform = pre_transform  # hàm (PIL, row) -> PIL, dùng cho test độ bền / kiểm tra thiên lệch
        self.cache = cache if (cache is not None and len(cache)) else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = self.cache.get(r["path"]) if self.cache is not None else None
        if img is None:
            try:
                img = Image.open(r["path"]).convert("RGB")
            except Exception as e:  # ảnh hỏng: thay bằng ảnh đen để không dừng cả lần chạy
                log.warning(f"Lỗi đọc {r['path']}: {e}")
                img = Image.new("RGB", (self.crop, self.crop))
        if self.pre_transform is not None:
            img = self.pre_transform(img, r)
        if self.train and self.aug_prob > 0:
            img = random_degrade(img, self.aug_prob)
        w, h = img.size
        if min(w, h) < self.crop and self.mode != "full":
            img = TF.pad(img, [0, 0, max(0, self.crop - w), max(0, self.crop - h)], padding_mode="reflect")
        if self.mode == "full":  # toàn ảnh: không cắt; ảnh 224x224 thu nhỏ CHỈ dùng cho CLIP (ngữ nghĩa)
            if self.train and random.random() < 0.5:
                img = TF.hflip(img)
            g224 = TF.to_tensor(img.resize((224, 224), Image.BICUBIC))
            x = TF.to_tensor(pad_full(img, self.crop))
            return x, g224, torch.tensor(r["label"], dtype=torch.float32), torch.tensor(r["env"]), torch.tensor(i)
        x = TF.to_tensor(img)
        if self.train and self.train_tiles:
            if random.random() < 0.5:
                x = x.flip(-1)
            H, W = x.shape[1], x.shape[2]
            k = min(self.train_tiles, len(tile_positions(H, W, self.crop)))  # ảnh 256x256 chỉ có 1 mảnh
            pos = [(random.randint(0, H - self.crop), random.randint(0, W - self.crop)) for _ in range(k)]
            tiles = torch.stack([x[:, t:t + self.crop, l:l + self.crop] for t, l in pos])
            return (tiles, TF.center_crop(x, [self.crop, self.crop]), torch.tensor(r["label"], dtype=torch.float32),
                    torch.tensor(r["env"]), torch.tensor(i))
        if self.train:
            t = random.randint(0, x.shape[1] - self.crop)
            l = random.randint(0, x.shape[2] - self.crop)
            x = x[:, t:t + self.crop, l:l + self.crop]
            if random.random() < 0.5:
                x = x.flip(-1)
        elif self.mode == "tiles":
            pos = tile_positions(x.shape[1], x.shape[2], self.crop, self.max_tiles)
            tiles = torch.stack([x[:, t:t + self.crop, l:l + self.crop] for t, l in pos])
            return (tiles, TF.center_crop(x, [self.crop, self.crop]), torch.tensor(r["label"], dtype=torch.float32),
                    torch.tensor(r["env"]), torch.tensor(i))
        else:
            x = TF.center_crop(x, [self.crop, self.crop])
        return x, torch.tensor(r["label"], dtype=torch.float32), torch.tensor(r["env"]), torch.tensor(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make-lists", "build-debiased", "stats", "cache-crops", "cut-list"])
    ap.add_argument("--n", type=int, help="cut-list: số ảnh tối đa mỗi (nguồn, nhãn)")
    ap.add_argument("--out", help="cut-list: tên danh sách mới")
    ap.add_argument("--lists", nargs="*", default=[])
    ap.add_argument("--fakes-only", action="store_true", help="build-debiased: chỉ nén ảnh giả (cho tập test)")
    ap.add_argument("--source", choices=["genimage", "forensynths", "folder", "benchmarks", "realpool"])
    ap.add_argument("--name")
    ap.add_argument("--root")
    ap.add_argument("--list")
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    if a.cmd == "make-lists":
        if a.source == "genimage":
            lists_genimage(cfg)
        elif a.source == "forensynths":
            lists_forensynths(cfg)
        elif a.source == "folder":
            lists_folder(cfg, a.name, a.root)
        elif a.source == "benchmarks":
            for n, r in cfg["test_benchmarks"].items():
                lists_folder(cfg, n, r)
        elif a.source == "realpool":
            lists_realpool(cfg)
    elif a.cmd == "cut-list":
        cut_list(cfg, a.list, a.n, a.out)
    elif a.cmd == "build-debiased":
        build_debiased(cfg, a.list, fakes_only=a.fakes_only)
    elif a.cmd == "stats":
        stats(cfg, a.list)
    elif a.cmd == "cache-crops":
        for n in (a.lists or [a.list]):
            if (lists_dir(cfg) / f"{n}.csv").exists():
                build_crop_cache(cfg, n)
            else:
                log.warning(f"Bỏ qua cache {n}: chưa có lists/{n}.csv")


if __name__ == "__main__":
    main()
