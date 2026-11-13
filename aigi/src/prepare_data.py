"""Chuẩn bị dữ liệu: TẢI (nguồn miễn phí, không cần đăng nhập) -> GIẢI NÉN -> SẮP XẾP -> KIỂM TRA -> ghi config.local.yaml.

Chạy trên máy của bạn (Windows / Linux), trong terminal VS Code:
  python run_all.py --prepare-data                   # = python -m src.prepare_data --plan core
  python -m src.prepare_data --plan minimal          # bản nhỏ (~40 GB) để chạy thử toàn quy trình trên dữ liệu thật
  python -m src.prepare_data --plan core             # đủ cho bài báo (~300-350 GB lúc cao điểm)
  python -m src.prepare_data --plan full             # thêm benchmark phụ lục (rất lớn)
  python -m src.prepare_data --list                  # xem từng mục, dung lượng, nguồn
  python -m src.prepare_data --only synthbuster coco_train   # chỉ vài mục
  python -m src.prepare_data --root E:/aigi-data     # đổi nơi lưu (nên là ổ SSD lớn)

Chạy lại an toàn: mục đã xong được ghi trong <root>/_prepare_status.json và bỏ qua; tải dở được tải tiếp.
Sau khi giải nén thành công, file nén do script này tải về sẽ bị xoá để tiết kiệm ổ (giữ lại: --keep-archives).

Cần: pip install huggingface_hub datasets pyarrow requests tqdm   (đã có trong requirements.txt)
      7-Zip cho GenImage dạng zip nhiều phần (.z01...): Windows: winget install 7zip.7zip | Ubuntu: sudo apt install p7zip-full

Nguồn (đều miễn phí, KHÔNG cần đăng nhập; giấy phép phần lớn chỉ cho nghiên cứu phi thương mại — xem bảng LIST):
  GenImage          HF jzousz/GenImage (bản sao không chính thức của bộ gốc, zip nhiều phần)  | HF nebula/GenImage-arrow
  COCO 2017         images.cocodataset.org (ảnh thật cho mô phỏng)
  Synthbuster       Zenodo 10066460 (+ ảnh thật RAISE-1k từ grip.unina.it)
  OpenFake          HF ComplexDataLab/OpenFake (bộ sinh 2024-2025: FLUX, SD3.5, DALL-E 3, Midjourney 6/7, Imagen...)
  Phụ lục           HF sywang/CNNDetection (ProGAN), UnivFD (Google Drive qua gdown), HF Junwei-Xi/EvalGEN
  Chameleon         KHÔNG tải tự động được (phải gửi email xin tác giả AIDE) -> tuỳ chọn
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

from .config import PROJECT_ROOT, deep_merge, load_config
from .utils import get_logger

log = get_logger("prepare")

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Tên viết tắt dùng trong code <- thư mục trong GenImage (theo README chính thức)
GENIMAGE = {
    "sd14": "stable_diffusion_v_1_4", "sd15": "stable_diffusion_v_1_5", "mj": "Midjourney", "adm": "ADM",
    "glide": "glide", "wukong": "wukong", "vqdm": "VQDM", "biggan": "BigGAN",
}
GENIMAGE_GB = {"sd14": 96.4, "sd15": 98.5, "mj": 227.5, "adm": 38.8, "glide": 32.3, "wukong": 97.2, "vqdm": 38.2,
               "biggan": 24.2}
# Gợi ý nhận diện tên bộ sinh trong bản arrow (cột generator) -> tên viết tắt
GEN_ALIASES = {
    "sd14": ["sd14", "sdv14", "sdv4", "stablediffusionv14", "stablediffusion14", "stablediffusionv4"],
    "sd15": ["sd15", "sdv15", "sdv5", "stablediffusionv15", "stablediffusion15"],
    "mj": ["mj", "midjourney"], "adm": ["adm"], "glide": ["glide"], "wukong": ["wukong"], "vqdm": ["vqdm"],
    "biggan": ["biggan"],
}

ITEMS = {
    # tên: (mô tả, GB tải về, GB sau giải nén, plans, giấy phép/ghi chú)
    "genimage_sd14": ("GenImage SD1.4 đầy đủ (train + test) — dữ liệu TRAIN theo giao thức chuẩn", 96.4, 100,
                      ["core", "full"], "HF jzousz/GenImage; giấy phép theo bộ GenImage gốc (nghiên cứu)"),
    "genimage_test": ("GenImage: tập test của 7 bộ sinh còn lại (SD1.5, MJ, Wukong, ADM, GLIDE, VQDM, BigGAN)", 15, 15,
                      ["core", "lite"], "HF nebula/GenImage-arrow (CC BY-NC-SA 4.0); --genimage-test zip để lấy bản zip gốc ~557 GB"),
    "genimage_mini": ("GenImage bản nhỏ: 20k ảnh train SD1.4 + test mọi bộ sinh (từ bản arrow)", 20, 20,
                      ["minimal"], "HF nebula/GenImage-arrow (CC BY-NC-SA 4.0)"),
    "genimage_all_zip": ("GenImage đủ 8 bộ sinh dạng zip gốc (chỉ cho --plan full)", 653, 660, ["full"],
                         "HF jzousz/GenImage"),
    "coco_train": ("COCO train2017 — ảnh thật cho mô phỏng", 19, 19, ["core", "full"], "COCO (giấy phép từng ảnh, Flickr)"),
    "genimage_sd14_fit": ("GenImage SD1.4 cho ổ nhỏ: lọc + nén NGAY KHI TẢI -> bản gốc (12k thật + 12k giả PNG) và "
                          "bản sạch Grommelt (thật JPEG q96 cạnh 450-550 + giả nén q96)", 9, 9, ["fit75"],
                          "HF nebula/GenImage-arrow; lưu lượng tải ~30-50 GB nhưng ổ chỉ giữ ~9 GB"),
    "genimage_test_fit": ("GenImage TEST cho ổ nhỏ: 2.000 thật + 2.000 giả mỗi bộ sinh (seed cố định), byte ảnh gốc",
                          8, 8, ["fit75"], "HF nebula/GenImage-arrow"),
    "genimage_sd14_lite": ("GenImage SD1.4: 1/3 ảnh TRAIN (shard chia đều, giữ byte ảnh gốc) + đủ ảnh TEST", 40, 40,
                           ["lite"], "HF nebula/GenImage-arrow (CC BY-NC-SA 4.0); đổi tỉ lệ bằng --train-frac"),
    "progan_lite": ("Giao thức 2 — TRAIN: ProGAN đúng 4 lớp (car/cat/chair/horse), không phải tải đủ 20 lớp", 15, 15,
                    ["lite_p2"], "HF jzousz/AIGCDetect_testset (four_train.zip, bản sao không chính thức)"),
    "coco_val": ("COCO val2017 — ảnh thật cho mô phỏng (bản nhỏ)", 1, 1, ["minimal", "lite"], "COCO"),
    "synthbuster": ("Synthbuster: 9 bộ sinh (DALL-E 2/3, Firefly, MJ v5, SD1.3/1.4/2/XL, GLIDE) + RAISE-1k thật",
                    14, 14, ["minimal", "core", "lite", "full"], "Zenodo 10066460 CC BY-NC-SA 4.0; RAISE: nghiên cứu"),
    "openfake": ("OpenFake (bộ sinh 2024-2025), lấy mẫu N ảnh / bộ sinh + ảnh thật", 8, 8, ["core", "lite", "fit75", "full"],
                 "HF ComplexDataLab/OpenFake CC BY-NC 4.0"),
    "chameleon": ("Chameleon (AIDE) — TUỲ CHỌN, tải tay", 0, 0, [], "Gửi email tattoo.ysl@gmail.com để xin; chỉ học thuật"),
    "progan": ("Giao thức 2 — TRAIN: ForenSynths ProGAN (dùng 4 lớp car/cat/chair/horse)", 75, 80, ["core_p2", "full"],
               "HF sywang/CNNDetection CC BY-NC-SA 4.0"),
    "forensynths_test": ("Giao thức 2 — TEST: ForenSynths (ProGAN, StyleGAN/2, BigGAN, CycleGAN, StarGAN, GauGAN, "
                         "Deepfake, CRN, IMLE, SAN, SITD, WFIR)", 20, 20, ["core_p2", "lite_p2", "full"],
                         "HF sywang/CNNDetection (CNN_synth_testset.zip) CC BY-NC-SA 4.0"),
    "univfd": ("Giao thức 2 — TEST: UnivFD diffusion (Guided, LDM x3, Glide x3, DALL-E)", 6, 6, ["core_p2", "lite_p2", "full"],
               "Google Drive (gdown) — bản 1k ảnh/tập"),
    "aigcdetect": ("Giao thức 2 — TEST: AIGCDetectBenchmark (16 tập con)", 33, 35, ["core_p2", "lite_p2", "full"],
                   "HF jzousz/AIGCDetect_testset (bản sao không chính thức, Apache-2.0) — tự kiểm tra cấu trúc"),
    "evalgen": ("DDA EvalGEN: Flux, GoT, Infinity, OmniGen, NOVA (phụ lục; chỉ ảnh giả)", 15, 15, ["full"],
                "HF Junwei-Xi/EvalGEN Apache-2.0"),
}


# =============================================================================
# tiện ích
# =============================================================================
def free_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def find_7z():
    for c in ("7z", "7za", "7zz", r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"):
        p = shutil.which(c) or (c if Path(c).exists() else None)
        if p:
            return p
    return None


def need_7z():
    z = find_7z()
    if not z:
        raise RuntimeError("Cần 7-Zip để giải nén zip nhiều phần. Windows: mở PowerShell chạy `winget install 7zip.7zip` "
                           "(hoặc tải ở 7-zip.org), Ubuntu/WSL: `sudo apt install p7zip-full`, rồi chạy lại.")
    return z


def extract(archive: Path, out: Path):
    """zip thường: Python; zip nhiều phần (.z01) / .7z.001: 7-Zip."""
    out.mkdir(parents=True, exist_ok=True)
    multi = archive.with_suffix(".z01").exists() or archive.name.endswith(".001") or archive.suffix == ".7z"
    log.info(f"Giải nén {archive.name} -> {out}")
    if not multi and archive.suffix == ".zip":
        try:
            with zipfile.ZipFile(archive) as z:
                z.extractall(out)
            return
        except (zipfile.BadZipFile, NotImplementedError) as e:
            log.warning(f"zipfile không mở được ({e}) -> thử 7-Zip")
    r = subprocess.run([need_7z(), "x", "-y", f"-o{out}", str(archive)])
    if r.returncode != 0:
        raise RuntimeError(f"Giải nén lỗi: {archive}")


def md5sum(p: Path, chunk=1 << 24):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def http_download(url: str, dst: Path, md5: str | None = None):
    """Tải có tiếp tục (Range) — mất mạng giữa chừng thì chạy lại sẽ tải tiếp."""
    import requests
    from tqdm import tqdm
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and (md5 is None or md5sum(dst) == md5):
        log.info(f"Đã có {dst.name}"); return dst
    part = dst.with_suffix(dst.suffix + ".part")
    pos = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with requests.get(url, stream=True, headers=headers, timeout=60, allow_redirects=True) as r:
        if r.status_code == 416:
            pos = 0
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + (pos if r.status_code == 206 else 0)
        mode = "ab" if r.status_code == 206 else "wb"
        with open(part, mode) as f, tqdm(total=total or None, initial=pos if mode == "ab" else 0, unit="B",
                                         unit_scale=True, desc=dst.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk); bar.update(len(chunk))
    part.replace(dst)
    if md5 and md5sum(dst) != md5:
        raise RuntimeError(f"MD5 sai cho {dst.name} — xoá file rồi tải lại")
    return dst


def hf_download(repo_id: str, local_dir: Path, patterns: list[str] | None = None, repo_type="dataset"):
    from huggingface_hub import snapshot_download
    log.info(f"Tải HF {repo_id} {patterns or ''} -> {local_dir}")
    return Path(snapshot_download(repo_id=repo_id, repo_type=repo_type, allow_patterns=patterns,
                                  local_dir=str(local_dir), max_workers=8))


def count_images(d: Path) -> int:
    return sum(1 for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXT) if Path(d).exists() else 0


def norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def save_bytes(img_field, out_dir: Path, stem: str) -> Path | None:
    """Ghi NGUYÊN byte ảnh gốc (không giải mã/nén lại) để giữ dấu vết; đuôi file theo định dạng thật."""
    from PIL import Image
    b = img_field.get("bytes") if isinstance(img_field, dict) else None
    if b is None and isinstance(img_field, dict) and img_field.get("path") and Path(img_field["path"]).exists():
        b = Path(img_field["path"]).read_bytes()
    if b is None:
        return None
    try:
        fmt = (Image.open(io.BytesIO(b)).format or "png").lower()
    except Exception:
        return None
    ext = {"jpeg": ".jpg", "mpo": ".jpg"}.get(fmt, f".{fmt}")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{stem}{ext}"
    p.write_bytes(b)
    return p


# =============================================================================
# từng mục dữ liệu
# =============================================================================
class Prep:
    def __init__(self, cfg, root: Path, keep_archives=False, genimage_test="auto", openfake_n=500, mini_train=20000,
                 train_frac=0.34, fit_n_raw=12000, fit_n_clean=12000, fit_n_test=2000, openfake_total=None):
        self.cfg, self.root, self.keep = cfg, root, keep_archives
        self.train_frac = train_frac
        self.fit_n_raw, self.fit_n_clean, self.fit_n_test = fit_n_raw, fit_n_clean, fit_n_test
        self.openfake_total = openfake_total
        self.dl = root / "_downloads"
        self.genimage_test, self.openfake_n, self.mini_train = genimage_test, openfake_n, mini_train
        self.status_f = root / "_prepare_status.json"
        self.status = json.loads(self.status_f.read_text(encoding="utf-8")) if self.status_f.exists() else {}
        self.local = {}  # sẽ ghi vào config.local.yaml

    # -------------------------------------------------- trạng thái
    def done(self, name, **info):
        self.status[name] = dict(done=True, **info)
        self.root.mkdir(parents=True, exist_ok=True)
        self.status_f.write_text(json.dumps(self.status, indent=2, ensure_ascii=False), encoding="utf-8")

    def cleanup(self, *paths):
        if self.keep:
            return
        for p in paths:
            p = Path(p)
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()

    # -------------------------------------------------- GenImage (zip gốc qua HF mirror)
    def _genimage_zip(self, short: str):
        top = GENIMAGE[short]
        dest = self.root / "raw" / "genimage"
        found = self._find_genimage_dir(dest, short)
        if found:
            log.info(f"GenImage {short} đã có ở {found}"); return found
        need = GENIMAGE_GB[short] * 2.1
        if free_gb(self.root) < need:
            raise RuntimeError(f"Không đủ ổ trống cho GenImage {short}: cần ~{need:.0f} GB, còn {free_gb(self.root):.0f} GB")
        need_7z()
        src = hf_download("jzousz/GenImage", self.dl / "genimage", [f"{top}/*"])
        mains = [p for p in (src / top).rglob("*.zip")]
        if not mains:
            raise RuntimeError(f"Không thấy file .zip trong {src / top}")
        for z in mains:
            extract(z, dest / top)
        self.cleanup(src / top)
        found = self._find_genimage_dir(dest, short)
        if not found:
            raise RuntimeError(f"Giải nén xong nhưng không thấy train/ai trong {dest / top}")
        return found

    @staticmethod
    def _find_genimage_dir(dest: Path, short: str):
        base = dest / GENIMAGE[short]
        if not base.exists():
            return None
        for d in [base, *sorted(p for p in base.rglob("*") if p.is_dir())]:
            if (d / "val" / "ai").exists() or (d / "train" / "ai").exists():
                return d
        return None

    def item_genimage_sd14(self):
        d = self._genimage_zip("sd14")
        self._set_generator("sd14", d)
        n = {s: count_images(d / s) for s in ("train", "val")}
        return dict(path=str(d), images=n)

    def item_genimage_all_zip(self):
        out = {}
        for short in GENIMAGE:
            d = self._genimage_zip(short)
            self._set_generator(short, d)
            out[short] = str(d)
        return out

    def _set_generator(self, short, d: Path):
        g = self.local.setdefault("genimage", {})
        g["root"] = str((self.root / "raw" / "genimage").as_posix())
        rel = Path(d).relative_to(self.root / "raw" / "genimage").as_posix()
        g.setdefault("generators", {})[short] = rel

    # -------------------------------------------------- GenImage (bản arrow: chỉ tải phần cần)
    # nebula/GenImage-arrow: data/{train,test}/<thư mục bộ sinh>/data-XXXXX-of-YYYYY.arrow (save_to_disk);
    # cột image = byte ảnh GỐC (PNG/JPEG), label 0 = thật / 1 = giả. Ảnh thật nằm cùng thư mục bộ sinh như bản gốc.
    ARROW_REPO = "nebula/GenImage-arrow"

    def _arrow_files(self, split: str, gen_dir: str):
        from huggingface_hub import HfApi
        files = [f for f in HfApi().list_repo_files(self.ARROW_REPO, repo_type="dataset")
                 if f.startswith(f"data/{split}/{gen_dir}/") and f.endswith(".arrow")]
        return sorted(files)

    @staticmethod
    def _iter_arrow(path: Path):
        import pyarrow as pa
        try:
            reader = pa.ipc.open_stream(pa.memory_map(str(path)))
        except Exception:
            reader = pa.ipc.open_file(pa.memory_map(str(path)))
            for i in range(reader.num_record_batches):
                yield from reader.get_batch(i).to_pylist()
            return
        for batch in reader:
            yield from batch.to_pylist()

    def _arrow_split(self, split_in: str, split_out: str, short: str, cap_per_class: int | None, frac: float = 1.0):
        """Tải từng shard của một bộ sinh, ghi ảnh gốc ra <root>/raw/genimage/_arrow/<short>/<split_out>/{ai,nature}.
        cap_per_class: dừng khi mỗi nhãn đủ số ảnh (bản minimal) — chỉ tải số shard cần thiết."""
        from huggingface_hub import hf_hub_download
        gen_dir = GENIMAGE[short]
        out = self.root / "raw" / "genimage" / "_arrow" / short
        cnt = {0: count_images(out / split_out / "nature"), 1: count_images(out / split_out / "ai")}
        files = self._arrow_files(split_in, gen_dir)
        if not files:
            raise RuntimeError(f"Không thấy shard data/{split_in}/{gen_dir}/ trong {self.ARROW_REPO}")
        if cap_per_class:
            random.Random(0).shuffle(files)  # lấy shard ngẫu nhiên để có cả ảnh thật lẫn giả
        if frac < 1.0:  # lấy shard CHIA ĐỀU trên toàn bộ danh sách (phủ cả phần ảnh thật lẫn ảnh giả nếu shard có thứ tự)
            k = max(1, round(len(files) * frac))
            idx = sorted({round(i * (len(files) - 1) / max(1, k - 1)) for i in range(k)})
            files = [files[i] for i in idx]
            log.info(f"  {short}/{split_in}: dùng {len(files)} shard (tỉ lệ {frac:.2f})")
        for k, f in enumerate(files):
            if cap_per_class and min(cnt.values()) >= cap_per_class:
                break
            marker = out / f".done_{split_in}_{Path(f).stem}"
            if marker.exists():
                continue
            local = Path(hf_hub_download(self.ARROW_REPO, f, repo_type="dataset", local_dir=str(self.dl / "genimage_arrow")))
            for j, r in enumerate(self._iter_arrow(local)):
                lab = int(r.get("label", 0))
                if cap_per_class and cnt[lab] >= cap_per_class:
                    continue
                img = r.get("image")
                img = img if isinstance(img, dict) else {"bytes": img}
                stem = Path(str(r.get("image_path") or f"{Path(f).stem}_{j:06d}")).stem
                if save_bytes(img, out / split_out / ("ai" if lab else "nature"), stem):
                    cnt[lab] += 1
            marker.parent.mkdir(parents=True, exist_ok=True); marker.write_text("ok")
            self.cleanup(local)
            log.info(f"  {short}/{split_in}: shard {k + 1}/{len(files)} -> thật {cnt[0]}, giả {cnt[1]}")
        self._set_generator(short, out)
        return cnt

    def _check_formats(self, out_root: Path):
        """In định dạng ảnh để kiểm tra (GenImage gốc: ảnh thật JPEG, ảnh giả phần lớn PNG)."""
        from collections import Counter
        for d in sorted(p for p in out_root.iterdir() if p.is_dir()):
            for cls in ("ai", "nature"):
                files = [p for p in d.rglob("*") if p.parent.name == cls][:300]
                if files:
                    log.info(f"  định dạng {d.name}/{cls}: {dict(Counter(p.suffix.lower() for p in files))}")

    def item_genimage_test(self):
        mode = self.genimage_test
        others = [g for g in GENIMAGE if g != "sd14"]
        if mode == "auto":
            need = sum(GENIMAGE_GB[g] for g in others) * 2.1
            mode = "zip" if free_gb(self.root) > need + 200 else "arrow"
            log.info(f"genimage_test: chọn '{mode}' (ổ trống {free_gb(self.root):.0f} GB, bản zip cần ~{need:.0f} GB)")
        if mode == "zip":
            for g in others:
                self._set_generator(g, self._genimage_zip(g))
            return dict(mode="zip")
        has_sd14 = bool(self.local.get("genimage", {}).get("generators", {}).get("sd14"))
        counts = {g: self._arrow_split("test", "val", g, None) for g in others + ([] if has_sd14 else ["sd14"])}
        self._check_formats(self.root / "raw" / "genimage" / "_arrow")
        return dict(mode="arrow", counts=counts)

    def item_genimage_sd14_lite(self):
        tr = self._arrow_split("train", "train", "sd14", None, frac=self.train_frac)
        te = self._arrow_split("test", "val", "sd14", None)
        log.info(f"SD1.4 lite: train thật {tr[0]}, giả {tr[1]} | test thật {te[0]}, giả {te[1]}")
        return dict(train=tr, test=te, frac=self.train_frac)

    # -------------------------------------------------- ổ nhỏ (--plan fit75): lọc + nén ngay khi tải
    def item_genimage_sd14_fit(self):
        """Duyệt shard train SD1.4 theo thứ tự ngẫu nhiên cố định; với mỗi ảnh:
          - bản GỐC  : giữ n_raw ảnh thật (byte gốc) + n_raw ảnh giả (PNG gốc)  -> _arrow/sd14/train/{nature,ai}
          - bản SẠCH : ảnh thật đạt chuẩn Grommelt (JPEG q96, cạnh 450-550) + ảnh giả NÉN JPEG q96
                       -> _clean/sd14/train/{nature,ai}  (ảnh thật trùng bản gốc dùng hard link, không tốn thêm ổ)
        Dừng khi đủ chỉ tiêu; mỗi shard tải xong là xoá."""
        from huggingface_hub import hf_hub_download
        from PIL import Image
        from .data import grommelt_real_ok
        d = self.cfg["debias"]
        n_raw, n_clean = self.fit_n_raw, self.fit_n_clean
        raw = self.root / "raw" / "genimage" / "_arrow" / "sd14" / "train"
        cln = self.root / "raw" / "genimage" / "_clean" / "sd14" / "train"
        cnt = dict(raw_real=count_images(raw / "nature"), raw_fake=count_images(raw / "ai"),
                   clean_real=count_images(cln / "nature"), clean_fake=count_images(cln / "ai"))
        done = lambda: (cnt["raw_real"] >= n_raw and cnt["raw_fake"] >= n_raw and cnt["clean_real"] >= n_clean
                        and cnt["clean_fake"] >= min(n_clean, max(cnt["clean_real"], 1)))
        files = self._arrow_files("train", GENIMAGE["sd14"])
        random.Random(0).shuffle(files)
        seen_real = 0
        for k, f in enumerate(files):
            if done():
                break
            marker = cln.parent / f".done_fit_{Path(f).stem}"
            if marker.exists():
                continue
            local = Path(hf_hub_download(self.ARROW_REPO, f, repo_type="dataset", local_dir=str(self.dl / "fit")))
            for j, r in enumerate(self._iter_arrow(local)):
                lab = int(r.get("label", 0))
                img = r.get("image"); b = img.get("bytes") if isinstance(img, dict) else img
                if b is None:
                    continue
                stem = Path(str(r.get("image_path") or f"{Path(f).stem}_{j:06d}")).stem
                if lab == 0:
                    seen_real += 1
                    p_raw = None
                    if cnt["raw_real"] < n_raw:
                        p_raw = save_bytes({"bytes": b}, raw / "nature", stem); cnt["raw_real"] += bool(p_raw)
                    if cnt["clean_real"] < n_clean:
                        try:
                            ok = grommelt_real_ok(Image.open(io.BytesIO(b)), d)
                        except Exception:
                            ok = False
                        if ok:
                            dst = cln / "nature" / f"{stem}.jpg"; dst.parent.mkdir(parents=True, exist_ok=True)
                            try:
                                os.link(p_raw, dst) if p_raw else dst.write_bytes(b)
                            except OSError:
                                dst.write_bytes(b)
                            cnt["clean_real"] += 1
                else:
                    if cnt["raw_fake"] < n_raw:
                        cnt["raw_fake"] += bool(save_bytes({"bytes": b}, raw / "ai", stem))
                    # ảnh giả bản sạch: giữ bằng số ảnh thật sạch (cân bằng lớp), nén JPEG q96 từ ảnh gốc
                    if cnt["clean_fake"] < min(n_clean, cnt["clean_real"] + 200):
                        try:
                            dst = cln / "ai" / f"{stem}.jpg"; dst.parent.mkdir(parents=True, exist_ok=True)
                            Image.open(io.BytesIO(b)).convert("RGB").save(dst, "JPEG", quality=d.get("jpeg_quality", 96))
                            cnt["clean_fake"] += 1
                        except Exception:
                            pass
            marker.parent.mkdir(parents=True, exist_ok=True); marker.write_text("ok")
            self.cleanup(local)
            log.info(f"  shard {k + 1}/{len(files)} | gốc: thật {cnt['raw_real']}, giả {cnt['raw_fake']} | sạch: thật "
                     f"{cnt['clean_real']}, giả {cnt['clean_fake']} | tỉ lệ ảnh thật đạt chuẩn Grommelt "
                     f"{cnt['clean_real'] / max(1, seen_real):.1%} | ổ trống {free_gb(self.root):.0f} GB")
        self._set_generator("sd14", raw.parent)
        g = self.local.setdefault("genimage", {})
        g["clean_train_dir"] = "_clean/sd14/train"
        # trên ổ có thể còn nhiều ảnh hơn (lần tải trước với N_TRAIN lớn hơn) -> danh sách chỉ lấy đúng N_TRAIN/lớp
        g["max_train_per_class"], g["max_clean_per_class"] = int(n_raw), int(n_clean)
        if cnt["clean_real"] < n_clean:
            log.warning(f"Chỉ tìm được {cnt['clean_real']} ảnh thật đạt chuẩn Grommelt (mục tiêu {n_clean}).")
        return cnt

    def item_genimage_test_fit(self):
        counts = {g: self._arrow_split("test", "val", g, self.fit_n_test) for g in GENIMAGE}
        self._check_formats(self.root / "raw" / "genimage" / "_arrow")
        return counts

    def item_progan_lite(self):
        dest = self.root / "raw" / "forensynths"
        classes = self.cfg["forensynths"]["classes"]
        if not all((dest / "train" / c).exists() for c in classes):
            src = hf_download("jzousz/AIGCDetect_testset", self.dl / "aigcdetect_train", ["four_train.zip"])
            z = next(src.rglob("four_train.zip"))
            tmp = self.dl / "four_train_x"; extract(z, tmp)
            (dest / "train").mkdir(parents=True, exist_ok=True)
            for c in classes:
                d = next((x for x in tmp.rglob(c) if x.is_dir() and (x / "0_real").exists() and (x / "1_fake").exists()), None)
                if d is None:
                    raise RuntimeError(f"Không thấy thư mục lớp '{c}' có 0_real/1_fake trong four_train.zip — xem {tmp}")
                shutil.move(str(d), str(dest / "train" / c))
            self.cleanup(tmp, z)
        self.local.setdefault("forensynths", {})["root"] = str(dest.as_posix())
        return {c: count_images(dest / "train" / c) for c in classes}

    def item_genimage_mini(self):
        n = self.mini_train // 2
        counts = {"sd14_train": self._arrow_split("train", "train", "sd14", n)}
        counts.update({g: self._arrow_split("test", "val", g, None) for g in GENIMAGE})
        self._check_formats(self.root / "raw" / "genimage" / "_arrow")
        return counts

    # -------------------------------------------------- COCO
    def _coco(self, split):
        dest = self.root / "real_pool" / "coco"
        if count_images(dest / split) > 1000:
            log.info(f"COCO {split} đã có")
        else:
            z = http_download(f"http://images.cocodataset.org/zips/{split}.zip", self.dl / f"{split}.zip")
            extract(z, dest)
            self.cleanup(z)
        self.local.setdefault("real_pool", {})["coco"] = str((dest / split).as_posix())
        return dict(images=count_images(dest / split))

    def item_coco_train(self):
        return self._coco("train2017")

    def item_coco_val(self):
        return self._coco("val2017")

    # -------------------------------------------------- Synthbuster + RAISE-1k
    def item_synthbuster(self):
        dest = self.root / "raw" / "test" / "synthbuster"
        if not any(dest.glob("*/fake")):
            z = http_download("https://zenodo.org/records/10066460/files/synthbuster.zip?download=1",
                              self.dl / "synthbuster.zip", md5="0695bd328e16ea21c5c9cc2ae1d994ff")
            tmp = self.dl / "synthbuster_x"
            extract(z, tmp)
            # mỗi thư mục con chứa ảnh = một bộ sinh -> dest/<bộ sinh>/fake/
            for d in sorted({p.parent for p in tmp.rglob("*") if p.suffix.lower() in IMG_EXT}):
                tgt = dest / d.name / "fake"; tgt.mkdir(parents=True, exist_ok=True)
                for f in d.iterdir():
                    if f.suffix.lower() in IMG_EXT:
                        shutil.move(str(f), str(tgt / f.name))
            self.cleanup(tmp, z)
        real = dest / "_real" / "real"
        if count_images(real) < 900:
            try:
                z = http_download("https://www.grip.unina.it/download/prog/DMimageDetection/real_RAISE_1k.zip",
                                  self.dl / "real_RAISE_1k.zip")
                tmp = self.dl / "raise_x"; extract(z, tmp)
                real.mkdir(parents=True, exist_ok=True)
                for f in tmp.rglob("*"):
                    if f.suffix.lower() in IMG_EXT:
                        shutil.move(str(f), str(real / f.name))
                self.cleanup(tmp, z)
            except Exception as e:
                log.warning(f"Không tải được RAISE-1k ({e}). Synthbuster chỉ có ảnh giả -> cần ảnh thật RAISE-1k: "
                            f"tải ở http://loki.disi.unitn.it/RAISE/ (bản 1k) rồi chép vào {real}")
        self.local.setdefault("test_benchmarks", {})["synthbuster"] = str(dest.as_posix())
        return dict(fake=count_images(dest) - count_images(real), real=count_images(real))

    # -------------------------------------------------- OpenFake (lấy mẫu, streaming)
    def item_openfake(self):
        from datasets import Image as HFImage, load_dataset
        dest = self.root / "raw" / "test" / "openfake"
        n = self.openfake_n
        if count_images(dest) >= n * 5:
            log.info("OpenFake đã có")
        else:
            ds = load_dataset("ComplexDataLab/OpenFake", "core", split="test", streaming=True)
            cols = ds.column_names or []
            img_col = next((c for c in ("image", "img", "jpg") if c in cols), "image")
            ds = ds.cast_column(img_col, HFImage(decode=False)).shuffle(seed=0, buffer_size=2000)
            cnt, n_real_cap = {}, n * 10
            for i, r in enumerate(ds):
                lab = str(r.get("label", "")).lower()
                is_fake = lab in ("fake", "1", "true", "synthetic")
                if is_fake and str(r.get("type", "image")).lower() not in ("image", "none", ""):
                    continue  # bỏ khung hình từ mô hình VIDEO (Sora, Wan...) — đề tài là ảnh
                model = norm(r.get("model", r.get("generator", "unknown"))) or "unknown"
                key = model if is_fake else "_real"
                cap = n if is_fake else n_real_cap
                if cnt.get(key, 0) >= cap:
                    continue
                if self.openfake_total:  # giới hạn tổng (ổ nhỏ)
                    n_f = sum(v for k2, v in cnt.items() if k2 != "_real")
                    if (is_fake and n_f >= self.openfake_total) or (not is_fake and cnt.get("_real", 0) >= self.openfake_total):
                        if n_f >= self.openfake_total and cnt.get("_real", 0) >= self.openfake_total:
                            break
                        continue
                sub = dest / key / ("fake" if is_fake else "real")
                if save_bytes(r[img_col], sub, f"of_{i:08d}"):
                    cnt[key] = cnt.get(key, 0) + 1
                if i % 5000 == 0:
                    log.info(f"  OpenFake: duyệt {i}, đã lưu {sum(cnt.values())} ({len(cnt)} nhóm)")
                if i > 400000:
                    break
            log.info(f"OpenFake: {cnt}")
        self.local.setdefault("test_benchmarks", {})["openfake"] = str(dest.as_posix())
        return dict(images=count_images(dest))

    # -------------------------------------------------- giao thức 2
    def _flatten_benchmark(self, src_root: Path, dest: Path):
        """Chuyển mọi thư mục có 0_real/1_fake (hoặc real/fake) về dest/<tập con>[/<lớp>]/{0_real,1_fake}."""
        names = {"0_real", "1_fake", "real", "fake"}
        moved = 0
        for d in sorted({p.parent for p in src_root.rglob("*") if p.is_dir() and p.name in names}):
            rel = d.relative_to(src_root)
            parts = [x for x in rel.parts if x.lower() not in ("test", "test_set", "testset", "cnn_synth_testset")]
            tgt = dest.joinpath(*parts) if parts else dest
            tgt.mkdir(parents=True, exist_ok=True)
            for sub in d.iterdir():
                if sub.is_dir() and sub.name in names and not (tgt / sub.name).exists():
                    shutil.move(str(sub), str(tgt / sub.name)); moved += 1
        return moved

    def item_forensynths_test(self):
        dest = self.root / "raw" / "test" / "forensynths"
        if count_images(dest) < 1000:
            src = hf_download("sywang/CNNDetection", self.dl / "cnndetection", ["CNN_synth_testset.zip"])
            z = next(src.rglob("CNN_synth_testset.zip"))
            tmp = self.dl / "cnnsynth_x"; extract(z, tmp)
            self._flatten_benchmark(tmp, dest)
            self.cleanup(tmp, z)
        self.local.setdefault("test_benchmarks", {})["forensynths"] = str(dest.as_posix())
        return dict(images=count_images(dest), subsets=sorted(p.name for p in dest.iterdir() if p.is_dir()))

    def item_aigcdetect(self):
        dest = self.root / "raw" / "test" / "aigcdetect"
        if count_images(dest) < 1000:
            src = hf_download("jzousz/AIGCDetect_testset", self.dl / "aigcdetect", ["test-large.zip"])
            z = next(src.rglob("test-large.zip"))
            tmp = self.dl / "aigcdetect_x"; extract(z, tmp)
            self._flatten_benchmark(tmp, dest)
            self.cleanup(tmp, z)
        subs = sorted(p.name for p in dest.iterdir() if p.is_dir())
        log.info(f"AIGCDetect: tập con tìm thấy {subs} — đối chiếu với 16 tập chuẩn (ProGAN, StyleGAN, BigGAN, CycleGAN, "
                 "StarGAN, GauGAN, StyleGAN2, WFIR, ADM, Glide, Midjourney, SD v1.4, SD v1.5, VQDM, Wukong, DALLE2)")
        self.local.setdefault("test_benchmarks", {})["aigcdetect"] = str(dest.as_posix())
        return dict(images=count_images(dest), subsets=subs)

    # -------------------------------------------------- phụ lục
    def item_progan(self):
        dest = self.root / "raw" / "forensynths"
        if not (dest / "train").exists():
            src = hf_download("sywang/CNNDetection", self.dl / "cnndetection", ["progan_train.7z.*"])
            first = sorted(src.rglob("progan_train.7z.001"))[0]
            extract(first, self.dl / "progan_x")
            inner = next((self.dl / "progan_x").rglob("progan_train.zip"), None)
            if inner:
                extract(inner, self.dl / "progan_y")
            base = next(p for p in (self.dl / "progan_y", self.dl / "progan_x") if p.exists())
            train = next((d for d in base.rglob("*") if d.is_dir() and (d / "car").exists()), None)
            if train is None:
                raise RuntimeError("Không tìm thấy thư mục lớp (car, cat, ...) sau khi giải nén ProGAN")
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(train), str(dest / "train"))
            self.cleanup(src, self.dl / "progan_x", self.dl / "progan_y")
        self.local.setdefault("forensynths", {})["root"] = str(dest.as_posix())
        return dict(images=count_images(dest))

    def item_univfd(self):
        dest = self.root / "raw" / "test" / "univfd"
        if count_images(dest) < 1000:
            import gdown
            z = self.dl / "univfd_diffusion.zip"
            if not z.exists():
                gdown.download(id="1FXlGIRh_Ud3cScMgSVDbEWmPDmjcrm1t", output=str(z), quiet=False)
            tmp = self.dl / "univfd_x"; extract(z, tmp)
            base = next((d for d in tmp.rglob("*") if d.is_dir() and (d / "laion").exists()), tmp)
            # cặp chuẩn của UnivFD: guided <-> imagenet, các bộ còn lại <-> laion
            for d in base.iterdir():
                if not d.is_dir() or d.name in ("laion", "imagenet"):
                    continue
                real_src = base / ("imagenet" if d.name == "guided" else "laion")
                (dest / d.name).mkdir(parents=True, exist_ok=True)
                fake_dir = next((x for x in d.rglob("*") if x.is_dir() and x.name in ("1_fake", "fake")), d)
                real_dir = next((x for x in real_src.rglob("*") if x.is_dir() and x.name in ("0_real", "real")), real_src)
                shutil.copytree(fake_dir, dest / d.name / "fake", dirs_exist_ok=True)
                shutil.copytree(real_dir, dest / d.name / "real", dirs_exist_ok=True)
            self.cleanup(tmp, z)
        self.local.setdefault("test_benchmarks", {})["univfd"] = str(dest.as_posix())
        return dict(images=count_images(dest))

    def item_evalgen(self):
        dest = self.root / "raw" / "test" / "evalgen"
        if count_images(dest) < 1000:
            src = hf_download("Junwei-Xi/EvalGEN", self.dl / "evalgen", ["*.zip"])
            tmp = self.dl / "evalgen_x"
            for z in src.rglob("*.zip"):
                extract(z, tmp)
            for d in sorted({p.parent for p in tmp.rglob("*") if p.suffix.lower() in IMG_EXT}):
                tgt = dest / d.name / "fake"; tgt.mkdir(parents=True, exist_ok=True)
                for f in d.iterdir():
                    if f.suffix.lower() in IMG_EXT:
                        shutil.move(str(f), str(tgt / f.name))
            self.cleanup(tmp, src)
        log.warning("EvalGEN chỉ có ảnh giả: cần ghép ảnh thật (ví dụ COCO val2017) vào evalgen/_real/real — "
                    "chỉ dùng cho phụ lục vì ảnh thật khác nguồn có thể gây thiên lệch.")
        self.local.setdefault("test_benchmarks", {})["evalgen"] = str(dest.as_posix())
        return dict(images=count_images(dest))

    def item_chameleon(self):
        dest = self.root / "raw" / "test" / "chameleon"
        if count_images(dest) > 0:
            self.local.setdefault("test_benchmarks", {})["chameleon"] = str(dest.as_posix())
            return dict(images=count_images(dest))
        raise RuntimeError(f"Chameleon không tải tự động được: gửi email cho tác giả AIDE (tattoo.ysl@gmail.com) xin bộ "
                           f"dữ liệu, rồi giải nén vào {dest} (dạng 0_real/ 1_fake/). Bài chính vẫn chạy được khi thiếu.")

    # -------------------------------------------------- ghi cấu hình
    def write_local_config(self, cfg_path: Path):
        p = cfg_path.with_name(cfg_path.stem + ".local.yaml")
        cur = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
        cur = cur or {}
        base = {"paths": {"data_root": str(self.root.as_posix()), "cache": str((self.root / "cache").as_posix()),
                          "runs": str((self.root / "runs").as_posix())}}
        deep_merge(cur, base)
        deep_merge(cur, self.local)
        # ảnh ImageNet thật trong train SD1.4 cũng là kho ảnh thật cho mô phỏng
        g = cur.get("genimage", {})
        if g.get("generators", {}).get("sd14"):
            sd = Path(g["root"]) / g["generators"]["sd14"] / "train" / "nature"
            if sd.exists():
                cur.setdefault("real_pool", {})["imagenet"] = str(sd.as_posix())
        p.write_text("# TỰ ĐỘNG tạo bởi src.prepare_data — đường dẫn dữ liệu trên máy này (ghi đè config.yaml)\n"
                     + yaml.safe_dump(cur, allow_unicode=True, sort_keys=False), encoding="utf-8")
        log.info(f"Đã ghi {p}")
        return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", choices=["fit75", "minimal", "lite", "lite_p2", "core", "core_p2", "full"], default="core",
                    help="lite: giao thức 1, 1/3 dữ liệu train (~90 GB) | lite_p2: + giao thức 2 (~170 GB) | "
                         "core: giao thức 1 đủ dữ liệu | core_p2: + giao thức 2 (~450 GB lúc cao điểm) | full: + phụ lục")
    ap.add_argument("--train-frac", type=float, default=0.34, help="tỉ lệ dữ liệu TRAIN GenImage SD1.4 cho plan lite")
    ap.add_argument("--skip", nargs="*", default=[], choices=list(ITEMS), help="bỏ qua các mục này (ví dụ aigcdetect)")
    ap.add_argument("--fit-n-raw", type=int, default=12000, help="fit75: số ảnh MỖI LỚP của bản gốc")
    ap.add_argument("--fit-n-clean", type=int, default=12000, help="fit75: số ảnh MỖI LỚP của bản sạch (Grommelt)")
    ap.add_argument("--fit-n-test", type=int, default=2000, help="fit75: số ảnh test MỖI LỚP mỗi bộ sinh")
    ap.add_argument("--usage", action="store_true", help="in dung lượng từng thư mục dữ liệu / kết quả")
    ap.add_argument("--only", nargs="*", choices=list(ITEMS))
    ap.add_argument("--root", help="nơi lưu dữ liệu (mặc định paths.data_root trong config)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--keep-archives", action="store_true")
    ap.add_argument("--genimage-test", choices=["auto", "arrow", "zip"], default="auto",
                    help="auto: zip gốc nếu ổ đủ (~1,2 TB), không thì bản arrow (chỉ tải phần test)")
    ap.add_argument("--openfake-n", type=int, default=500, help="số ảnh giả mỗi bộ sinh OpenFake")
    ap.add_argument("--mini-train", type=int, default=20000, help="số ảnh train SD1.4 cho --plan minimal")
    ap.add_argument("--force", nargs="*", default=[], help="làm lại các mục này dù đã đánh dấu xong")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg_path = Path(a.config) if a.config else PROJECT_ROOT / "config.yaml"
    cfg = load_config(cfg_path)
    root = Path(a.root or cfg["paths"]["data_root"])
    includes = {"core_p2": ["core"], "lite_p2": ["lite"]}.get(a.plan, [])
    names = a.only or [k for k, v in ITEMS.items() if a.plan in v[3] or any(p in v[3] for p in includes)]
    names = [n for n in names if n not in a.skip]
    if a.plan.startswith("lite") and a.genimage_test == "auto":
        a.genimage_test = "arrow"  # lite: chỉ tải phần test của GenImage

    if a.usage:
        def du(p):
            return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e9 if Path(p).exists() else 0
        rows = [(str(p), du(p)) for p in sorted(root.iterdir())] if root.exists() else []
        hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        rows.append((f"{hf} (mô hình tải từ Hugging Face)", du(hf)))
        for name, gb in rows:
            print(f"{gb:8.1f} GB  {name}")
        print(f"{sum(g for _, g in rows):8.1f} GB  TỔNG | ổ trống còn {free_gb(root):.0f} GB")
        return
    if a.list:
        print(f"{'mục':18} {'tải GB':>7} {'đĩa GB':>7}  plans               mô tả / nguồn")
        for k, (desc, dl, disk, plans, lic) in ITEMS.items():
            mark = "*" if k in names else " "
            print(f"{mark}{k:17} {dl:7.0f} {disk:7.0f}  {','.join(plans) or 'tay':18}  {desc}\n{'':38}{lic}")
        print(f"\n* = thuộc --plan {a.plan}. Dung lượng xấp xỉ; lúc giải nén cần thêm ~bằng dung lượng tải.")
        return

    total = sum(ITEMS[k][1] + ITEMS[k][2] for k in names)
    log.info(f"Kế hoạch '{a.plan}': {names}")
    log.info(f"Nơi lưu {root} | ổ trống {free_gb(root):.0f} GB | cần tối đa ~{total:.0f} GB (đã tính giải nén)")
    if free_gb(root) < total * 0.6:
        log.warning("Ổ có thể không đủ. Chọn ổ khác bằng --root, hoặc dùng --plan minimal trước.")
    if a.plan == "fit75":
        a.openfake_n = min(a.openfake_n, 100)
    prep = Prep(cfg, root, a.keep_archives, a.genimage_test, a.openfake_n, a.mini_train, a.train_frac,
                a.fit_n_raw, a.fit_n_clean, a.fit_n_test, 2000 if a.plan == "fit75" else None)
    if a.plan == "fit75":
        # cấu hình cho ổ nhỏ: ghi vào config.local.yaml
        deep_merge(prep.local, {"sim": {"n_per_family": 10000}, "cache": {"use": False},
                                "main": {"data": "debiased"}, "train": {"keep_last": False}})
    results = {}
    for name in names:
        if prep.status.get(name, {}).get("done") and name not in a.force:
            log.info(f"[{name}] đã xong trước đó -> bỏ qua")
            prev = prep.status[name]
            for k in ("local",):
                if prev.get(k):
                    deep_merge(prep.local, prev[k])
            results[name] = "đã có"
            continue
        log.info(f"\n===== [{name}] {ITEMS[name][0]} =====")
        before = json.dumps(prep.local, sort_keys=True)
        try:
            info = getattr(prep, f"item_{name}")()
            delta = json.loads(json.dumps(prep.local))  # lưu phần cấu hình mục này đóng góp để lần sau dựng lại
            prep.done(name, info=info, local=delta)
            results[name] = f"xong {info}"
        except Exception as e:
            log.error(f"[{name}] LỖI: {e}")
            results[name] = f"LỖI: {e}"
        _ = before
    p = prep.write_local_config(cfg_path)
    print("\n" + "=" * 80 + "\nTỔNG KẾT CHUẨN BỊ DỮ LIỆU")
    for k, v in results.items():
        print(f"  {k:18} {v}")
    print(f"\nĐường dẫn đã ghi vào {p}. Tiếp theo: python run_all.py --check  rồi  python run_all.py")


if __name__ == "__main__":
    main()
