"""Bảng so sánh với các bài báo khác: gộp số CHẠY THẬT (từ dự đoán từng ảnh) với số ĐÃ CÔNG BỐ (reported/*.csv).

Mỗi giao thức dùng đúng tập con, đúng độ đo, đúng cách tính như các bài báo:
  genimage      : train SD1.4 -> test 8 bộ sinh GenImage, Acc (ngưỡng 0,5) và AP mỗi bộ sinh, trung bình 8 cột
  forensynths8  : train ProGAN 4 lớp -> 8 GAN của ForenSynths
  ojha8         : train ProGAN 4 lớp -> 8 tập diffusion của UnivFD (Ojha)
  aigcdetect16  : train ProGAN -> 16 tập con AIGCDetectBenchmark
  univfd19      : train ProGAN -> 19 tập con (ForenSynths 11 + Ojha 8)

Ký hiệu trong bảng: (không dấu) = chạy thật trên CÙNG ảnh test; † = checkpoint của tác giả train trên dữ liệu
riêng (không theo giao thức); ‡ = số chép từ bài báo (không có dự đoán từng ảnh -> không kiểm định DeLong).

  python -m src.compare                     # mọi giao thức có dữ liệu
  python -m src.compare --protocol genimage
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import re
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT, load_config
from .utils import get_logger

log = get_logger("compare")
REPORTED = PROJECT_ROOT / "reported"

FS11 = [("ProGAN", "progan"), ("CycleGAN", "cyclegan"), ("BigGAN", "biggan"), ("StyleGAN", "stylegan"),
        ("GauGAN", "gaugan"), ("StarGAN", "stargan"), ("Deepfake", "deepfake"), ("SITD", "seeingdark"),
        ("SAN", "san"), ("CRN", "crn"), ("IMLE", "imle")]
OJHA = [("DALLE", "dalle"), ("Glide_100_10", "glide_100_10"), ("Glide_100_27", "glide_100_27"),
        ("Glide_50_27", "glide_50_27"), ("ADM", "guided"), ("LDM_100", "ldm_100"), ("LDM_200", "ldm_200"),
        ("LDM_200_cfg", "ldm_200_cfg")]

# cột: (tên cột như trong bài báo, danh sách test, bí danh nhóm nguồn)
PROTOCOLS = {
    "genimage": dict(
        reported="genimage_sd14.csv", runs=["M_final_s*", "M_erm_s*", "V_final_s*", "B_*_s*", "EXT_*"],
        cols=[(c, "genimage_all_val", a) for c, a in [
            ("Midjourney", "mj"), ("SDv1.4", "sd14"), ("SDv1.5", "sd15"), ("ADM", "adm"), ("GLIDE", "glide"),
            ("Wukong", "wukong"), ("VQDM", "vqdm"), ("BigGAN", "biggan")]]),
    "forensynths8": dict(
        reported="progan_forensynths8.csv", runs=["P_final_s*", "P_erm_s*", "EXT_*"],
        cols=[(c, "test_forensynths", a) for c, a in [
            ("ProGAN", "progan"), ("StyleGAN", "stylegan"), ("StyleGAN2", "stylegan2"), ("BigGAN", "biggan"),
            ("CycleGAN", "cyclegan"), ("StarGAN", "stargan"), ("GauGAN", "gaugan"), ("DeepFake", "deepfake")]]),
    "ojha8": dict(
        reported="progan_ojha8.csv", runs=["P_final_s*", "P_erm_s*", "EXT_*"],
        cols=[(c, "test_univfd", a) for c, a in OJHA]),
    "aigcdetect16": dict(
        reported="progan_aigcdetect16.csv", runs=["P_final_s*", "P_erm_s*", "EXT_*"],
        cols=[(c, "test_aigcdetect", a) for c, a in [
            ("ProGAN", "progan"), ("StyleGAN", "stylegan"), ("BigGAN", "biggan"), ("CycleGAN", "cyclegan"),
            ("StarGAN", "stargan"), ("GauGAN", "gaugan"), ("StyleGAN2", "stylegan2"), ("WFIR", "whichfaceisreal"),
            ("ADM", "adm"), ("Glide", "glide"), ("Midjourney", "midjourney"), ("SD v1.4", "stable_diffusion_v_1_4"),
            ("SD v1.5", "stable_diffusion_v_1_5"), ("VQDM", "vqdm"), ("Wukong", "wukong"), ("DALLE2", "dalle2")]]),
    "univfd19": dict(
        reported="progan_univfd19.csv", runs=["P_final_s*", "P_erm_s*", "EXT_*"],
        cols=[(c, "test_forensynths", a) for c, a in FS11] + [
            (c, "test_univfd", a) for c, a in [("Guided", "guided"), ("LDM_200", "ldm_200"),
                                               ("LDM_200_cfg", "ldm_200_cfg"), ("LDM_100", "ldm_100"),
                                               ("Glide_100_27", "glide_100_27"), ("Glide_50_27", "glide_50_27"),
                                               ("Glide_100_10", "glide_100_10"), ("DALL-E", "dalle")]]),
}

NAMES = {"M_final": "Ours (ConvNeXt + cổng)", "V_final": "Ours (ViT + cổng)", "M_erm": "ERM, 27 nhóm (cùng dữ liệu)",
         "B_cnnspot": "CNNSpot (train lại)", "B_univfd": "UnivFD (train lại)", "B_aligned": "Aligned (train lại)",
         "P_final": "Ours (ConvNeXt + cổng)", "P_erm": "ERM, 27 nhóm (cùng dữ liệu)"}
# Cách xử lý baseline ngoài (xem src/baselines/README.md): A = checkpoint đúng giao thức; C = dữ liệu riêng của tác giả
EXT_HANDLING = {"NPR": "A", "C2P-CLIP": "A", "DRCT": "A", "AIDE": "A", "Effort": "A", "OmniAID": "A", "VIB-Net": "A",
                "FreqNet": "B", "SAFE": "B", "CO-SPY": "B", "Aligned": "C", "BFree": "C", "B-Free": "C", "DDA": "C",
                "SPAI": "C", "RINE": "C"}


def norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def group_of(source: str, list_name: str) -> str:
    """genimage: 'mj' ; benchmark: 'univfd/guided/...' -> 'guided' (thành phần đầu sau tên benchmark)."""
    parts = str(source).split("/")
    if len(parts) > 1 and norm(parts[0]) == norm(list_name.replace("test_", "")):
        parts = parts[1:]
    return norm(parts[0])


def scores(run_dir: Path, cols, thr=0.5):
    """Acc@thr và AP mỗi cột từ preds_clean_<list>.npz. Trả về {cột: (acc, ap)} hoặc None nếu thiếu dữ liệu."""
    from .skfallback import average_precision_score
    out, cache = {}, {}
    for col, lst, alias in cols:
        if lst not in cache:
            f = run_dir / f"preds_clean_{lst}.npz"
            cache[lst] = np.load(f, allow_pickle=False) if f.exists() else None
        d = cache[lst]
        if d is None:
            continue
        g = np.array([group_of(s, lst) for s in d["source"]])
        y, p = d["label"].astype(int), d["prob"].astype(float)
        m = g == norm(alias)
        if not (y[m] == 1).any():
            continue
        if not (y[m] == 0).any():
            m = m | (y == 0)  # tập con chỉ có ảnh giả: dùng ảnh thật chung của benchmark
        acc = float(((p[m] >= thr) == (y[m] == 1)).mean()) * 100
        ap = float(average_precision_score(y[m], p[m])) * 100
        out[col] = (acc, ap)
    return out


def measured_rows(cfg, proto, min_cols=0.5):
    root = Path(cfg["paths"]["runs"])
    runs = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
    picked = [r for r in runs if any(fnmatch.fnmatch(r, pat) for pat in proto["runs"])]
    groups = {}
    for r in picked:
        sc = scores(root / r, proto["cols"])
        if len(sc) < max(1, len(proto["cols"]) * min_cols):  # thiếu quá nửa cột -> không đưa vào bảng
            continue
        groups.setdefault(re.sub(r"_s\d+$", "", r), []).append(sc)
    rows = []
    for base, lst in groups.items():
        if base.startswith("EXT_"):
            nm = base[4:]
            h = EXT_HANDLING.get(nm, "C")
            label, mark = nm + (" (checkpoint gốc)" if h == "A" else " (code gốc, train lại)" if h == "B" else ""), \
                ("†" if h == "C" else "")
        else:
            label, mark = NAMES.get(base, base), ""
        rows.append(dict(method=label, mark=mark, kind="đo", seeds=len(lst), runs=lst))
    return rows


def build(cfg, name, out_dir: Path, min_cols=0.5):
    proto = PROTOCOLS[name]
    colnames = [c for c, _, _ in proto["cols"]]
    meas = measured_rows(cfg, proto, min_cols)
    rep_path = REPORTED / proto["reported"]
    rep = list(csv.DictReader(open(rep_path, encoding="utf-8"))) if rep_path.exists() else []
    if not meas:
        log.info(f"[{name}] chưa có run nào có dự đoán trên các danh sách test của giao thức này -> bỏ qua")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for k, metric in enumerate(("Acc", "AP")):
        lines = []
        for r in meas:
            vals = []
            for c in colnames:
                xs = [s[c][k] for s in r["runs"] if c in s]
                vals.append(np.mean(xs) if xs else np.nan)
            means = [np.nanmean([s[c][k] for c in colnames if c in s]) for s in r["runs"]]
            sd = float(np.std(means)) if len(means) > 1 else None
            lines.append(dict(method=r["method"] + r["mark"], source="bài này (chạy thật)", n_seeds=r["seeds"],
                              **{c: v for c, v in zip(colnames, vals)}, Mean=float(np.mean(means)), Std=sd))
        for r in rep:
            if r.get("metric") != metric or str(r.get("canonical")) != "1":
                continue
            row = dict(method=r["method"] + "‡", source=f"{r['source_paper']} {r['table_id']}", n_seeds="")
            for c in colnames:
                key = next((h for h in r if norm(h) == norm(c)), None)
                row[c] = float(r[key]) if key and r[key] not in ("", None) else np.nan
            row["Mean"] = float(r["Mean"]) if r.get("Mean") else np.nan
            row["Std"] = None
            row["note"] = r.get("note", "")
            lines.append(row)
        if not lines:
            continue
        hdr = ["method", "source", "n_seeds"] + colnames + ["Mean", "Std", "note"]
        f = out_dir / f"compare_{name}_{metric.lower()}.csv"
        with open(f, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=hdr, extrasaction="ignore")
            w.writeheader()
            for ln in lines:
                w.writerow({k2: (f"{v:.2f}" if isinstance(v, float) and not np.isnan(v) else
                                 "" if isinstance(v, float) else v) for k2, v in ln.items()})
        # LaTeX: số tốt nhất mỗi cột in đậm (chỉ trong các dòng có số)
        tex = [r"\begin{tabular}{l" + "c" * (len(colnames) + 1) + "}", r"\toprule",
               "Method & " + " & ".join(colnames) + r" & Mean \\", r"\midrule"]
        best = {c: np.nanmax([ln[c] for ln in lines]) if any(not np.isnan(ln[c]) for ln in lines) else None
                for c in colnames + ["Mean"]}
        for i, ln in enumerate(lines):
            if i == len(meas):
                tex.append(r"\midrule")
            cells = []
            for c in colnames + ["Mean"]:
                v = ln[c]
                if isinstance(v, float) and not np.isnan(v):
                    s = f"{v:.1f}"
                    if c == "Mean" and ln.get("Std") is not None:
                        s += rf"$\pm${ln['Std']:.1f}"
                    cells.append(rf"\textbf{{{s}}}" if best[c] is not None and abs(v - best[c]) < 1e-9 else s)
                else:
                    cells.append("-")
            mname = ln["method"].replace("_", r"\_").replace("‡", r"$^\ddagger$").replace("†", r"$^\dagger$")
            tex.append(mname + " & " + " & ".join(cells) + r" \\")
        tex += [r"\bottomrule", r"\end{tabular}",
                "% †: checkpoint của tác giả train trên dữ liệu riêng; ‡: số do tác giả báo cáo (nguồn ở file .csv cùng tên)"]
        (out_dir / f"compare_{name}_{metric.lower()}.tex").write_text("\n".join(tex), encoding="utf-8")
        written.append(f.name)
    log.info(f"[{name}] -> {written}")
    return written


def build_drop(cfg, out_dir: Path):
    """Mỗi mô hình sụt bao nhiêu khi bỏ đường tắt JPEG: test GenImage gốc vs test sạch (ảnh giả nén JPEG q96).
    Mô hình học 'PNG = giả' sẽ sụt mạnh; mô hình học dấu vết thật sẽ sụt ít."""
    from .skfallback import roc_auc_score
    root = Path(cfg["paths"]["runs"])
    gens = PROTOCOLS["genimage"]["cols"]
    rows = {}
    for rd in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        res = {}
        for lst in ("genimage_all_val", "genimage_all_val_debiased"):
            f = rd / f"preds_clean_{lst}.npz"
            if not f.exists():
                break
            d = np.load(f, allow_pickle=False)
            g = np.array([group_of(x, lst) for x in d["source"]]); y = d["label"].astype(int); p = d["prob"].astype(float)
            accs, aucs = [], []
            for _, _, alias in gens:
                m = g == norm(alias)
                if not (y[m] == 1).any():
                    continue
                if not (y[m] == 0).any():
                    m = m | (y == 0)
                accs.append(((p[m] >= 0.5) == (y[m] == 1)).mean() * 100)
                aucs.append(roc_auc_score(y[m], p[m]) * 100)
            res[lst] = (np.mean(accs), np.mean(aucs)) if accs else None
        if len(res) == 2 and all(res.values()):
            rows.setdefault(re.sub(r"_s\d+$", "", rd.name), []).append(res)
    if not rows:
        log.info("[drop] chưa có run nào có dự đoán trên cả genimage_all_val và genimage_all_val_debiased"); return
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "drop_raw_vs_clean.csv"
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "n_seeds", "acc_test_goc", "acc_test_sach", "sut_acc", "auroc_test_goc", "auroc_test_sach", "sut_auroc"])
        for base, lst in sorted(rows.items()):
            a0 = np.mean([r["genimage_all_val"][0] for r in lst]); a1 = np.mean([r["genimage_all_val_debiased"][0] for r in lst])
            u0 = np.mean([r["genimage_all_val"][1] for r in lst]); u1 = np.mean([r["genimage_all_val_debiased"][1] for r in lst])
            w.writerow([base, len(lst), f"{a0:.2f}", f"{a1:.2f}", f"{a0 - a1:.2f}", f"{u0:.2f}", f"{u1:.2f}", f"{u0 - u1:.2f}"])
    log.info(f"[drop] -> {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=list(PROTOCOLS))
    ap.add_argument("--out", default="paper/tables")
    ap.add_argument("--min-cols", type=float, default=0.5, help="bỏ run có dự đoán cho ít hơn tỉ lệ cột này")
    ap.add_argument("--drop", action="store_true", help="chỉ dựng bảng sụt giảm test gốc -> test sạch")
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    if a.drop:
        build_drop(cfg, Path(a.out)); return
    for name in ([a.protocol] if a.protocol else list(PROTOCOLS)):
        build(cfg, name, Path(a.out), a.min_cols)


if __name__ == "__main__":
    main()
