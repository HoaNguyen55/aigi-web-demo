"""Bảng tổng hợp cho profile VIT (ba giai đoạn trên cùng thân ViT) — không chạy lại gì.

  python -m src.vit_report --out paper/vit

Đầu ra (paper/vit/):
  table_stages.csv/.md      Ba giai đoạn: ViT chỉ RGB -> ViT + từng đặc trưng -> ViT + tất cả (+ môi trường, + cổng + V-REx,
                            đối chứng RAND/IRM/GroupDRO). Cột: AUROC unseen test sạch/gốc, AP, Acc@0.5, Acc ngưỡng VAL, DROP.
  table_per_generator.csv   Mọi run ViT × test gốc/sạch × biến đổi ảnh × generator, 4 độ đo (dạng dài).
  table_auroc_wide.csv      AUROC theo generator (ảnh chưa biến đổi), hai khối test gốc/sạch.
  table_robust.csv          AUROC unseen theo biến đổi ảnh (JPEG/resize/...), test gốc/sạch.
  top3_per_generator.md     3 phương pháp cao nhất và ảnh hưởng theo từng generator.
Run thiếu thì bỏ dòng (ghi trong README.md), không lỗi.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .utils import get_logger

log = get_logger("vit_report")

STAGES = [
    ("1. ViT trước khi kết hợp đặc trưng", [
        ("VIT1_rgb_deb", "ViT chỉ RGB", "VIT1_rgb_raw"),
        ("VIT1_rgb_env", "ViT chỉ RGB + môi trường mô phỏng", None)]),
    ("2. ViT sau khi kết hợp TỪNG đặc trưng", [
        ("VIT2_wpt_deb", "ViT + wavelet 16 dải", "VIT2_wpt_raw"),
        ("VIT2_hh_deb", "ViT + HH", "VIT2_hh_raw"),
        ("VIT2_fft_deb", "ViT + phổ Fourier 8 vành", "VIT2_fft_raw"),
        ("VIT2_npr_deb", "ViT + NPR", "VIT2_npr_raw"),
        ("VIT2_clip_deb", "ViT + CLIP", "VIT2_clip_raw")]),
    ("3. ViT + TẤT CẢ đặc trưng", [
        ("VIT3_all_erm", "ViT + tất cả, ERM (không môi trường)", None),
        ("VIT3_all_erm_env", "ViT + tất cả, ERM + môi trường", None),
        ("VIT3_all_vrex_final", "ViT + tất cả + cổng + phạt V-REx", None),
        ("VIT3_all_rand", "đối chứng: chọn ngẫu nhiên cùng số nhóm", None),
        ("VIT3_all_irm", "đối chứng: phạt IRM", None),
        ("VIT3_all_dro", "đối chứng: GroupDRO", None)]),
]
CLEAN, RAW = "genimage_all_val_debiased", "genimage_all_val"


def main():
    global RAW, CLEAN
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/vit")
    ap.add_argument("--tables", default="paper/tables")
    ap.add_argument("--raw-list", default=RAW, help="danh sách test gốc (vitnhanh: genimage_all_val_s600)")
    ap.add_argument("--clean-list", default=CLEAN, help="danh sách test sạch")
    ap.add_argument("--config", default=None, help="(run_all.py tự thêm; báo cáo chỉ đọc paper/tables nên không dùng)")
    a = ap.parse_args()
    RAW, CLEAN = a.raw_list, a.clean_list
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f = Path(a.tables) / "all_results.csv"
    if not f.exists():
        raise SystemExit(f"Chưa có {f} — chạy: python -m src.evaluate tables --out {a.tables}")
    df = pd.read_csv(f)
    df["exp_base"] = df["exp"].str.replace(r"_s\d+$", "", regex=True)
    vit = df[df.exp_base.str.startswith("VIT")].copy()
    clean = vit[vit.variant == "clean"]
    metrics = [m for m in ("auroc", "ap", "acc05", "acc_vt") if m in vit.columns]
    notes = []

    def get(exp, lst, src, col="auroc"):
        for l in (lst, RAW):  # chưa có bản test sạch thì tạm lấy test gốc (ghi vào README)
            s = clean[(clean.exp_base == exp) & (clean.list == l) & (clean.source == src)][col]
            if len(s):
                if l != lst and "test sạch chưa có -> dùng test gốc" not in notes:
                    notes.append("test sạch chưa có -> dùng test gốc")
                return float(s.mean())
        return float("nan")

    # ---------- Bảng ba giai đoạn
    rows = []
    for stage, items in STAGES:
        for exp, name, raw_exp in items:
            if get(exp, CLEAN, "mean_unseen") != get(exp, CLEAN, "mean_unseen"):  # NaN
                notes.append(f"thiếu run {exp}"); continue
            r = {"Giai đoạn": stage, "Mô hình": name, "exp": exp,
                 "AUROC unseen (test sạch)": get(exp, CLEAN, "mean_unseen"),
                 "AUROC unseen (test gốc)": get(exp, RAW, "mean_unseen"),
                 "AUROC seen (test sạch)": get(exp, CLEAN, "mean_seen"),
                 "AP unseen": get(exp, CLEAN, "mean_unseen", "ap") if "ap" in metrics else float("nan"),
                 "Acc@0,5 unseen": get(exp, CLEAN, "mean_unseen", "acc05") if "acc05" in metrics else float("nan"),
                 "Acc ngưỡng VAL unseen": get(exp, CLEAN, "mean_unseen", "acc_vt") if "acc_vt" in metrics else float("nan")}
            if raw_exp:  # DROP = Acc test gốc − Acc test sạch của mô hình train trên dữ liệu GỐC
                r["Train gốc → test sạch (AUROC unseen)"] = get(raw_exp, CLEAN, "mean_unseen")
                r["DROP (Acc@0,5)"] = get(raw_exp, RAW, "mean", "acc05") - get(raw_exp, CLEAN, "mean", "acc05")
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(out / "table_stages.csv", index=False)
    if not t.empty:
        tf = t.copy()
        for c in tf.columns:
            if tf[c].dtype.kind == "f":
                tf[c] = (tf[c] * 100).round(1)
        md = tf.to_markdown(index=False) if hasattr(tf, "to_markdown") else tf.to_string(index=False)
        (out / "table_stages.md").write_text("# Ba giai đoạn trên thân ViT (×100)\n\n" + md, encoding="utf-8")

    # ---------- Theo generator (dạng dài + rộng) và độ bền
    gen_src = [c for c in vit.source.unique() if not str(c).startswith("mean") and str(c) != "all"]
    full = vit[vit.list.isin([RAW, CLEAN])].copy()
    full["test"] = full["list"].map({RAW: "gốc", CLEAN: "sạch"})
    t5 = full.groupby(["exp_base", "test", "variant", "source"])[metrics].mean().reset_index()
    t5.to_csv(out / "table_per_generator.csv", index=False)
    w = t5[(t5.variant == "clean") & (t5.source.isin(gen_src + ["mean_seen", "mean_unseen"]))]
    if not w.empty:
        (w.pivot_table(index="exp_base", columns=["test", "source"], values="auroc") * 100).round(1)\
            .to_csv(out / "table_auroc_wide.csv")
    rob = t5[(t5.source == "mean_unseen")]
    if rob.variant.nunique() > 1:
        (rob.pivot_table(index=["exp_base", "test"], columns="variant", values="auroc") * 100).round(1)\
            .to_csv(out / "table_robust.csv")
    else:
        notes.append("chưa có đánh giá độ bền (VIT_ROBUST)")

    # ---------- Top 3 và ảnh hưởng theo generator
    rank = clean[(clean.list == CLEAN) & (clean.source == "mean_unseen")].groupby("exp_base")["auroc"].mean()
    rank = rank.sort_values(ascending=False)
    top3 = list(rank.index[:3])
    if top3:
        lines = ["# 3 mô hình ViT cao nhất (AUROC unseen, test sạch) và ảnh hưởng theo generator", ""]
        for exp in top3:
            lines.append(f"## {exp}: {rank[exp]*100:.1f}")
            sub = t5[(t5.exp_base == exp) & (t5.variant == "clean") & (t5.source.isin(gen_src))]
            piv = sub.pivot_table(index="source", columns="test", values=metrics) * 100
            lines.append(piv.round(1).to_markdown() if hasattr(piv, "to_markdown") else piv.round(1).to_string())
            s_ = sub[sub.test == "sạch"].set_index("source")["auroc"]
            g_ = sub[sub.test == "gốc"].set_index("source")["auroc"]
            if len(s_):
                lines.append(f"- Test sạch: dễ nhất {s_.idxmax()} ({s_.max()*100:.1f}), khó nhất {s_.idxmin()} ({s_.min()*100:.1f}), "
                             f"chênh {(s_.max()-s_.min())*100:.1f} điểm.")
            d = ((g_ - s_).dropna() * 100) if len(s_) and len(g_) else pd.Series(dtype=float)
            if len(d):
                lines.append(f"- Test gốc − sạch: lớn nhất {d.idxmax()} ({d.max():+.1f}), nhỏ nhất {d.idxmin()} ({d.min():+.1f}); "
                             f"dương lớn = còn dựa vào đường tắt ở generator đó.")
            lines.append("")
        (out / "top3_per_generator.md").write_text("\n".join(lines), encoding="utf-8")

    readme = ["# Profile VIT — bảng tổng hợp", "", "table_stages: ba giai đoạn; table_per_generator / table_auroc_wide: theo generator;",
              "table_robust: ảnh đã biến đổi; top3_per_generator: 3 mô hình cao nhất.", ""]
    if notes:
        readme += ["## Thiếu", ""] + [f"- {n}" for n in notes]
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    log.info(f"Đã gom vào {out}" + (f" — thiếu {len(notes)} mục, xem README.md" if notes else ""))


if __name__ == "__main__":
    main()
