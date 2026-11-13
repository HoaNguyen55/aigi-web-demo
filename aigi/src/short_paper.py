"""Gom đúng bảng/hình cho BÀI HỘI NGHỊ RÚT GỌN (hạng B) từ kết quả đã có — không chạy lại gì.

  python -m src.short_paper --out paper/short

Đầu ra (paper/short/):
  table1_features.csv/.tex   5 họ đặc trưng: AUROC seen / unseen trên test sạch, và DROP (Acc gốc - Acc sạch)
  table2_main.csv/.tex       Phương pháp chính vs ERM, cổng không phạt, chọn ngẫu nhiên, CNNSpot, UnivFD (mean±std, 3 seed) + p DeLong
  table3_published.csv/.tex  Acc từng bộ sinh GenImage (train dữ liệu gốc) đặt cạnh số đã công bố ‡ (từ compare_genimage_acc.csv)
  table4_stability.csv       Jaccard / Spearman của cổng qua seed (nếu có)
  fig_gates.png, fig_robust.png (chép từ paper/figures nếu có)
  README.md                  ghi rõ bảng nào lấy từ file nào, thiếu gì
Các run không có thì bỏ dòng, không lỗi.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import pandas as pd

from .config import load_config
from .utils import get_logger

log = get_logger("short")

FAMILIES = [("rgb", "RGB"), ("wpt", "Wavelet"), ("fft", "Phổ (FFT)"), ("npr", "NPR"), ("clip", "CLIP")]
MAIN_ROWS = [("M_final", "Cổng + V-REx (đề xuất)"), ("M_erm", "ERM mọi nhóm"), ("A_nopen_final", "Cổng không phạt (β=0)"),
             ("RAND_final", "Chọn ngẫu nhiên cùng số nhóm"), ("B_cnnspot", "CNNSpot (train lại)"), ("B_univfd", "UnivFD (train lại)")]
DELONG_B = {"M_erm": "M_erm_s0", "A_nopen_final": None, "RAND_final": "RAND_final_s0", "B_cnnspot": "B_cnnspot_s0",
            "B_univfd": "B_univfd_s0"}


def _tex(df: pd.DataFrame, path: Path, caption: str):
    path.write_text("% " + caption + "\n" + df.to_latex(index=False, na_rep="-", escape=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/short")
    ap.add_argument("--tables", default="paper/tables")
    ap.add_argument("--figures", default="paper/figures")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out, tb = Path(a.out), Path(a.tables)
    out.mkdir(parents=True, exist_ok=True)
    notes = []

    allres = tb / "all_results.csv"
    if not allres.exists():
        raise SystemExit(f"Chưa có {allres} — chạy: python -m src.evaluate tables --out {tb}")
    df = pd.read_csv(allres)
    df["exp_base"] = df["exp"].str.replace(r"_s\d+$", "", regex=True)
    clean = df[df.variant == "clean"]

    # Tên run khác nhau giữa profile core và full -> thử lần lượt các tên tương đương
    ALIAS = {"M_final": ["M_final", "N4_b_final"], "M_erm": ["M_erm", "N4_b_erm"], "A_nopen_final": ["A_nopen_final", "E4_nogate"],
             "B_cnnspot": ["B_cnnspot", "S_resnet50"], "B_univfd": ["B_univfd", "S_clip_lp"], "B_aligned": ["B_aligned", "ALIGNED"],
             "N_envonly": ["N2_final", "N2_erm"]}
    for k in ("rgb", "wpt", "hh", "fft", "npr", "clip"):
        ALIAS[f"F_{k}_deb"] = [f"F_{k}_deb", f"E1_{k}_deb"]; ALIAS[f"F_{k}_raw"] = [f"F_{k}_raw", f"E1_{k}_raw"]
        ALIAS[f"FB_{k}_convnext"] = [f"FB_{k}_convnext", f"F_{k}_deb"]   # core: F_*_deb đã là ConvNeXt-Tiny

    def pick(exp_base, lst, source, col="auroc"):
        for e in ALIAS.get(exp_base, [exp_base]):
            for l in (lst, lst.replace("_debiased", "")):  # chưa có test sạch thì lấy test gốc (ghi chú trong README)
                s = clean[(clean.exp_base == e) & (clean.list == l) & (clean.source == source)][col]
                if len(s):
                    if l != lst and f"dùng {l} thay {lst}" not in notes:
                        notes.append(f"dùng {l} thay {lst}")
                    return (s.mean(), s.std(), len(s))
        return (float("nan"), float("nan"), 0)

    # ---------- Bảng 1: 5 họ đặc trưng
    rows = []
    for key, name in FAMILIES:
        seen = pick(f"F_{key}_deb", "genimage_all_val_debiased", "mean_seen")[0]
        unseen = pick(f"F_{key}_deb", "genimage_all_val_debiased", "mean_unseen")[0]
        acc_raw = pick(f"F_{key}_raw", "genimage_all_val", "mean", "acc05")[0]
        acc_cln = pick(f"F_{key}_raw", "genimage_all_val_debiased", "mean", "acc05")[0]
        unseen_rawtest = pick(f"F_{key}_deb", "genimage_all_val", "mean_unseen")[0]      # train sạch, test GỐC
        unseen_rawtrain = pick(f"F_{key}_raw", "genimage_all_val", "mean_unseen")[0]     # train gốc, test GỐC
        unseen_rawtrain_clean = pick(f"F_{key}_raw", "genimage_all_val_debiased", "mean_unseen")[0]  # train gốc, test sạch
        rows.append({"Họ đặc trưng": name, "AUROC seen": seen, "AUROC unseen": unseen, "Seen − unseen": seen - unseen,
                     "Unseen, train sạch → test gốc": unseen_rawtest,
                     "Unseen, train gốc → test gốc": unseen_rawtrain,
                     "Unseen, train gốc → test sạch": unseen_rawtrain_clean,
                     "DROP (Acc gốc − sạch, mô hình train gốc)": acc_raw - acc_cln})
    t1 = pd.DataFrame(rows)
    if t1.iloc[:, 1:].isna().all().all():
        notes.append("Bảng 1: không thấy run F_*_deb / F_*_raw — chưa chạy đợt 2?")
    t1.to_csv(out / "table1_features.csv", index=False)
    _tex((t1.set_index("Họ đặc trưng") * 100).round(1).reset_index(), out / "table1_features.tex",
         "Bảng 1: họ đặc trưng trên GenImage đã khử thiên lệch (train SD1.4, test 7 bộ sinh), ×100")

    # ---------- Bảng 1b: mỗi họ đặc trưng trên nhiều thân mạng (AUROC unseen, test sạch)
    rows = []
    for key, name in FAMILIES + [("hh", "HH")]:
        r = {"Họ đặc trưng": name}
        for tl, tag in (("genimage_all_val_debiased", "test sạch"), ("genimage_all_val", "test gốc")):
            r[f"cnn nhỏ ({tag})"] = pick(f"E1_{key}_deb", tl, "mean_unseen")[0]          # profile full: thân mặc định
            r[f"ConvNeXt-Tiny ({tag})"] = pick(f"FB_{key}_convnext", tl, "mean_unseen")[0]
            for b in ("resnet50", "vit", "cnn"):
                v = pick(f"FB_{key}_{b}", tl, "mean_unseen")[0]
                if b == "cnn" and v != v:
                    continue
                r[f"{b} ({tag})"] = v
        rows.append(r)
    t1b = pd.DataFrame(rows)
    if t1b.iloc[:, 1:].isna().all().all():
        notes.append("Bảng 1b: không thấy run FB_<họ>_<thân> — chưa chạy đợt 2b (nhiều thân mạng)?")
    else:
        t1b.to_csv(out / "table1b_backbones.csv", index=False)
        _tex((t1b.set_index("Họ đặc trưng") * 100).round(1).reset_index(), out / "table1b_backbones.tex",
             "Bảng 1b: mỗi họ đặc trưng trên 4 thân mạng, AUROC unseen trên test sạch, ×100")

    # ---------- Bảng 2: phương pháp chính vs đối chứng (3 seed) + DeLong
    runs_dir = Path(cfg["paths"]["runs"])
    rows = []
    for exp, name in MAIN_ROWS:
        m, s, n = pick(exp, "genimage_all_val_debiased", "mean_unseen")
        if n == 0:
            notes.append(f"Bảng 2: không có run {exp}"); continue
        ap_m = pick(exp, "genimage_all_val_debiased", "mean_unseen", "ap")[0]
        acc_m = pick(exp, "genimage_all_val_debiased", "mean", "acc_vt")[0]
        if acc_m != acc_m:  # chưa có ngưỡng VAL -> Acc@0.5
            acc_m = pick(exp, "genimage_all_val_debiased", "mean", "acc05")[0]
        p = ""
        rb = DELONG_B.get(exp)
        if rb:
            f = runs_dir / f"delong_M_final_s0_vs_{rb}_genimage_all_val.json"
            if f.exists():
                p = f"{json.load(open(f))['p_value']:.3g}"
        m_raw = pick(exp, "genimage_all_val", "mean_unseen")[0]   # cùng run, test GỐC (còn đường tắt)
        rows.append({"Phương pháp": name, "AUROC unseen": m, "± std": s, "seed": n, "AUROC unseen (test gốc)": m_raw,
                     "AP unseen": ap_m, "Acc (ngưỡng VAL)": acc_m, "p DeLong vs đề xuất": p})
    t2 = pd.DataFrame(rows)
    t2.to_csv(out / "table2_main.csv", index=False)
    if not t2.empty:
        t2f = t2.copy()
        for c in ("AUROC unseen", "± std", "AUROC unseen (test gốc)", "AP unseen", "Acc (ngưỡng VAL)"):
            t2f[c] = (t2f[c] * 100).round(1)
        _tex(t2f, out / "table2_main.tex", "Bảng 2: so với đối chứng trên bộ sinh chưa thấy, test sạch, ×100")

    # ---------- Bảng 3: so với số đã công bố (dữ liệu gốc)
    cmp = tb / "compare_genimage_acc.csv"
    if cmp.exists():
        c = pd.read_csv(cmp)
        keep = c[c.iloc[:, 0].astype(str).str.contains("M_final_raw|M_erm_raw|‡|dagger|EXT_|CNNSpot|UnivFD|NPR|FreqNet|AIDE|DRCT|Effort|C2P",
                                                      regex=True, na=False)] if len(c) else c
        (keep if len(keep) else c).to_csv(out / "table3_published.csv", index=False)
        _tex(keep if len(keep) else c, out / "table3_published.tex", "Bảng 3: Acc trên GenImage gốc, ‡ = số đã công bố, † = checkpoint gốc")
    else:
        notes.append("Bảng 3: chưa có compare_genimage_acc.csv — chạy bước COMPARE")

    # ---------- Bảng 4: ổn định cổng
    st = runs_dir / "analysis_gate_stability.json"
    if st.exists():
        d = json.load(open(st))
        pd.json_normalize(d).to_csv(out / "table4_stability.csv", index=False)
    else:
        notes.append("Bảng 4: chưa có analysis_gate_stability.json — chạy bước ANA_STAB (cần STAB)")

    # ---------- Bảng 5: ĐẦY ĐỦ mọi phương pháp × từng generator × test gốc/sạch × biến đổi ảnh, 4 độ đo
    # (một dòng = một run-base, một danh sách test, một biến thể ảnh, một generator; seed được lấy trung bình)
    gen_src = [c for c in df.source.unique() if not str(c).startswith("mean") and str(c) != "all"]
    full = df[df.list.isin(["genimage_all_val", "genimage_all_val_debiased"])].copy()
    full["test"] = full["list"].map({"genimage_all_val": "gốc", "genimage_all_val_debiased": "sạch"})
    metrics = [m for m in ("auroc", "ap", "acc05", "acc_vt") if m in full.columns]
    t5 = (full.groupby(["exp_base", "test", "variant", "source"])[metrics].mean().reset_index()
              .rename(columns={"exp_base": "Phương pháp", "test": "Test", "variant": "Biến đổi ảnh", "source": "Generator"}))
    t5.to_csv(out / "table5_per_generator_full.csv", index=False)
    # bản rộng dễ đọc: AUROC theo generator, ảnh chưa biến đổi (variant clean), hai cột test gốc/sạch
    wide = (t5[(t5["Biến đổi ảnh"] == "clean") & (t5["Generator"].isin(gen_src + ["mean_seen", "mean_unseen"]))]
            .pivot_table(index="Phương pháp", columns=["Test", "Generator"], values="auroc", aggfunc="mean"))
    (wide * 100).round(1).to_csv(out / "table5_auroc_wide.csv")
    for m in metrics:
        w = (t5[(t5["Biến đổi ảnh"] == "clean") & (t5["Generator"].isin(gen_src + ["mean_seen", "mean_unseen"]))]
             .pivot_table(index="Phương pháp", columns=["Test", "Generator"], values=m, aggfunc="mean"))
        (w * 100).round(1).to_csv(out / f"table5_{m}_wide.csv")
    # ảnh ĐÃ biến đổi (JPEG/resize/...): AUROC theo generator cho các run có đánh giá độ bền
    robv = t5[t5["Biến đổi ảnh"] != "clean"]
    if not robv.empty:
        (robv.pivot_table(index=["Phương pháp", "Test", "Biến đổi ảnh"], columns="Generator", values="auroc", aggfunc="mean") * 100)\
            .round(1).to_csv(out / "table5_robust_per_generator.csv")

    # ---------- Bảng 6: 3 phương pháp cao nhất (theo AUROC unseen, test sạch) và ảnh hưởng lên TỪNG generator
    rank = (clean[(clean.list == "genimage_all_val_debiased") & (clean.source == "mean_unseen")]
            .groupby("exp_base")["auroc"].mean().sort_values(ascending=False))
    rank = rank[~rank.index.str.startswith(("F_", "FB_", "E1_", "E2_"))] if len(rank) else rank  # chỉ xét phương pháp hoàn chỉnh
    top3 = list(rank.index[:3])
    if top3:
        lines = ["# 3 phương pháp cao nhất (AUROC unseen, test sạch) và ảnh hưởng theo từng generator", ""]
        for exp in top3:
            lines.append(f"## {exp}: unseen sạch = {rank[exp]*100:.1f}")
            sub = t5[(t5["Phương pháp"] == exp) & (t5["Biến đổi ảnh"] == "clean") & (t5["Generator"].isin(gen_src))]
            piv = sub.pivot_table(index="Generator", columns="Test", values=metrics, aggfunc="mean") * 100
            lines.append(piv.round(1).to_markdown() if hasattr(piv, "to_markdown") else piv.round(1).to_string())
            # generator khó/dễ nhất và chênh gốc-sạch
            a = sub[sub["Test"] == "sạch"].set_index("Generator")["auroc"]
            if len(a):
                lines.append(f"- Dễ nhất (test sạch): {a.idxmax()} ({a.max()*100:.1f}); khó nhất: {a.idxmin()} ({a.min()*100:.1f}); "
                             f"chênh {((a.max()-a.min())*100):.1f} điểm.")
            b = sub[sub["Test"] == "gốc"].set_index("Generator")["auroc"]
            if len(a) and len(b):
                d = (b - a).dropna() * 100
                if len(d):
                    lines.append(f"- Test gốc − test sạch theo generator: lớn nhất {d.idxmax()} ({d.max():+.1f}), "
                                 f"nhỏ nhất {d.idxmin()} ({d.min():+.1f}) → dương lớn = còn dựa vào đường tắt ở generator đó.")
            lines.append("")
        (out / "table6_top3_per_generator.md").write_text("\n".join(lines), encoding="utf-8")
        t5[t5["Phương pháp"].isin(top3)].to_csv(out / "table6_top3_per_generator.csv", index=False)
    else:
        notes.append("Bảng 6: chưa có run phương pháp hoàn chỉnh để xếp hạng")

    # ---------- Hình
    for pat, dst in (("*gate*", "fig_gates.png"), ("*robust*", "fig_robust.png")):
        fs = sorted(glob.glob(str(Path(a.figures) / pat)))
        if fs:
            shutil.copy(fs[0], out / dst)
        else:
            notes.append(f"Hình {dst}: không thấy {pat} trong {a.figures}")

    readme = ["# Bài hội nghị rút gọn — bảng/hình", "",
              "| File | Nguồn | Dùng ở mục |", "|---|---|---|",
              "| table1_features | all_results.csv (F_*_deb, F_*_raw) | 4.1 Họ đặc trưng nào tổng quát hoá |",
              "| table2_main | all_results.csv + delong_*.json | 4.2 Cổng + V-REx vs đối chứng |",
              "| table3_published | compare_genimage_acc.csv | 4.3 So với số đã công bố |",
              "| table4_stability | analysis_gate_stability.json | 4.4 Ổn định cổng |",
              "| table5_per_generator_full / table5_*_wide / table5_robust_per_generator | all_results.csv | mọi phương pháp × generator × test gốc/sạch × biến đổi, 4 độ đo |",
              "| table6_top3_per_generator (.md/.csv) | all_results.csv | 3 phương pháp cao nhất, ảnh hưởng theo từng generator |",
              "| fig_gates / fig_robust | paper/figures | 4.4 / 4.5 |", ""]
    if notes:
        readme += ["## Thiếu", ""] + [f"- {n}" for n in notes]
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    log.info(f"Đã gom vào {out}" + (f" — thiếu {len(notes)} mục, xem README.md" if notes else ""))


if __name__ == "__main__":
    main()
