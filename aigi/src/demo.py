"""Demo: ảnh kích thước bất kỳ -> THẬT hay AI, so sánh nhiều mô hình, kèm bản đồ điểm từng mảnh 256x256.

  python -m src.demo --images anh1.jpg thu_muc_anh/ --html demo.html     # tự chọn mô hình (ưu tiên bản MIL)
  python -m src.demo --runs VIT3_all_vrex_final_mil_s0 VIT2_clip_deb_s0 --images ...
  python -m src.demo --list-runs

Cách chấm: ảnh được chia thành các mảnh 256x256 phủ kín TOÀN ẢNH (không thu nhỏ, ảnh nhỏ hơn 256 được đệm phản chiếu).
- Mô hình MIL (run *_mil*): các mảnh của cùng ảnh được gộp bằng lớp chú ý đã học -> 1 điểm cho cả ảnh; trang HTML
  tô màu từng mảnh theo điểm riêng và viền đậm các mảnh mô hình chú ý nhiều nhất.
- Mô hình cũ (1 vùng giữa khi train): điểm các mảnh được gộp bằng trung bình logit (eval.tile_agg).
Ngưỡng quyết định: ngưỡng chọn trên VAL của run (chế độ toàn ảnh nếu đã chạy retune-thr). Không cần dữ liệu GenImage.
"""
from __future__ import annotations

import argparse
import base64
import gc
import html
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import load_config
from .evaluate import load_run, predict_detail
from .utils import get_logger

log = get_logger("demo")
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
# mô hình chính (ưu tiên bản MIL), rồi hai mốc so sánh
MAIN = ["VIT3_all_vrex_final_full", "VIT3_all_vrex_final_mil", "VIT3_all_vrex_final", "VIT3_all_erm_mil", "VIT3_all_erm", "M_final", "V_final"]
EXTRA = [["VIT2_clip_deb_full", "VIT2_clip_deb_mil", "VIT2_clip_deb"], ["VIT1_rgb_deb_full", "VIT1_rgb_deb_mil", "VIT1_rgb_deb"]]
NAMES = {"VIT3_all_vrex_final": "Kết hợp 2 hướng", "VIT3_all_erm": "Kết hợp 2 hướng (ERM)", "VIT2_clip_deb": "CLIP",
         "VIT1_rgb_deb": "RGB", "VIT1_rgb_raw": "RGB (dữ liệu gốc)", "VIT2_npr_deb": "NPR", "VIT2_hh_deb": "HH",
         "VIT2_wpt_deb": "WPT", "VIT2_fft_deb": "FFT"}


def collect_images(items):
    out = []
    for it in items:
        p = Path(it)
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.suffix.lower() in EXTS)
        elif p.suffix.lower() in EXTS:
            out.append(p)
        else:
            log.warning(f"Bỏ qua (không phải ảnh): {p}")
    return out


def available_runs(cfg):
    d = Path(cfg["paths"]["runs"])
    return sorted(p.name for p in d.iterdir() if (p / "final.pt").exists()) if d.exists() else []


def _find(have, base):
    for h in have:
        if h == base or (h.startswith(base + "_s") and h[len(base) + 2:].isdigit()):
            return h
    return None


def pick_runs(cfg, runs):
    have = available_runs(cfg)
    if runs:
        miss = [r for r in runs if r not in have]
        if miss:
            raise SystemExit(f"Không thấy final.pt của {miss}. Run đang có: {have}")
        return runs, have
    out = []
    for b in MAIN:
        if (r := _find(have, b)):
            out.append(r); break
    for alts in EXTRA:
        for b in alts:
            if (r := _find(have, b)):
                out.append(r); break
    if not out and have:
        out = [have[0]]
    if not out:
        raise SystemExit(f"Chưa có run nào có final.pt trong {cfg['paths']['runs']}")
    return out, have


def nice_name(run):
    base = run[:-3] if run[-3:-1] == "_s" else run
    tag = ""
    for suf, lab in (("_full", " · toàn ảnh"), ("_mil", " · MIL")):
        if base.endswith(suf):
            base, tag = base[:-len(suf)], lab
    return NAMES.get(base, base) + tag


def img_b64(path, max_w=560):
    im = Image.open(path).convert("RGB")
    s = min(1.0, max_w / im.width)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BICUBIC) if s < 1 else im
    b = io.BytesIO(); im.save(b, "JPEG", quality=88)
    return base64.b64encode(b.getvalue()).decode(), im.width, im.height, s


def _color(p):  # xanh (thật) -> đỏ (AI)
    r, g = int(46 + (192 - 46) * p), int(139 + (57 - 139) * p)
    return f"rgb({r},{g},{int(87 + (43 - 87) * p)})"


def overlay_svg(d, w, h, s, thr, crop):
    W, H = d["size"]
    rects = []
    attn = d["attn"]
    top = set(np.argsort(-attn)[:max(1, min(3, len(attn)))].tolist()) if attn is not None and len(attn) > 1 else set()
    op = 0.16 if d.get("mode") == "full" else 0.32  # cửa sổ chồng lấn -> tô nhạt hơn
    for k, ((y, x), p) in enumerate(zip(d["positions"], d["tile_probs"])):
        x0, y0 = min(x, W) * s, min(y, H) * s
        x1, y1 = min(x + crop, W) * s, min(y + crop, H) * s
        sw = 3 if k in top else 0.6
        title = f"Mảnh {k + 1}: điểm AI {p:.3f}" + (f" · chú ý {attn[k]:.2f}" if attn is not None else "")
        rects.append(f"<rect x='{x0:.1f}' y='{y0:.1f}' width='{max(1, x1 - x0):.1f}' height='{max(1, y1 - y0):.1f}' "
                     f"fill='{_color(float(p))}' fill-opacity='{op}' stroke='{'#111' if k in top else '#fff'}' "
                     f"stroke-width='{sw}'><title>{html.escape(title)}</title></rect>")
    return f"<svg class='ov' viewBox='0 0 {w} {h}' preserveAspectRatio='none'>{''.join(rects)}</svg>"


CSS = """
:root{--bg:#f4f6f9;--card:#fff;--ink:#1f2a44;--mut:#5b6577;--line:#dfe5ee;--ai:#c0392b;--real:#2e8b57;--navy:#17375e}
*{box-sizing:border-box}body{margin:0;font-family:Segoe UI,Calibri,Arial,sans-serif;background:var(--bg);color:var(--ink)}
header{background:var(--navy);color:#fff;padding:20px 28px}header h1{margin:0 0 6px;font-size:22px}
header p{margin:0;color:#c9d6ea;font-size:14px}.wrap{max-width:1180px;margin:0 auto;padding:20px}
.tools{display:flex;gap:18px;align-items:center;font-size:14px;color:var(--mut);margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:18px;
display:grid;grid-template-columns:minmax(0,560px) minmax(0,1fr);gap:20px}
@media(max-width:860px){.card{grid-template-columns:1fr}}
.pic{position:relative;line-height:0}.pic img{width:100%;border-radius:8px}.ov{position:absolute;inset:0;width:100%;height:100%}
body.nogrid .ov{display:none}.meta{font-size:12px;color:var(--mut);margin-top:8px;line-height:1.5;word-break:break-all}
.res h3{margin:0 0 10px;font-size:16px}.row{border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:10px}
.top{display:flex;justify-content:space-between;align-items:center}.nm{font-weight:600}
.chip{font-size:12px;font-weight:700;padding:3px 9px;border-radius:999px}.chip.ai{background:#f7e1de;color:var(--ai)}
.chip.real{background:#e0f1e7;color:var(--real)}.sc{font-size:22px;font-weight:700;margin:2px 0}
.bar{position:relative;height:10px;background:#e9edf3;border-radius:5px;margin:6px 0}.bar i{position:absolute;left:0;top:0;
bottom:0;border-radius:5px}.bar b{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--ink)}
.sub{font-size:12px;color:var(--mut);display:flex;justify-content:space-between}
.legend{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--mut)}
.grad{width:120px;height:10px;border-radius:5px;background:linear-gradient(90deg,rgb(46,139,87),rgb(192,57,43))}
footer{font-size:12px;color:var(--mut);padding:0 20px 30px;max-width:1180px;margin:0 auto;line-height:1.6}
"""


def write_html(out, models, paths, crop):
    h = [f"<!doctype html><html lang='vi'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>Demo phát hiện ảnh AI</title><style>{CSS}</style><body><header><h1>Phát hiện ảnh AI dựa trên đặc trưng hình ảnh</h1>"
         f"<p>Ảnh kích thước bất kỳ, không thu nhỏ. Mô hình toàn ảnh: đặc trưng tính trên cả ảnh, cửa sổ {crop}×{crop} trượt chồng lấn; "
         f"mô hình cũ: các mảnh {crop}×{crop} phủ kín ảnh. "
         f"Mô hình chính: <b>{html.escape(models[0]['name'])}</b>.</p></header><div class='wrap'>"
         "<div class='tools'><label><input type='checkbox' checked onchange=\"document.body.classList.toggle('nogrid',!this.checked)\"> "
         "Hiện điểm từng mảnh</label><span class='legend'>Thật<span class='grad'></span>AI</span>"
         "<span>Viền đen: 3 vùng mô hình chú ý nhiều nhất</span></div>"]
    for i, p in enumerate(paths):
        b64, w, hh, s = img_b64(p)
        d0 = models[0]["det"][i]
        W, H = d0["size"]
        h.append(f"<div class='card'><div><div class='pic' style='max-width:{max(w, 320)}px'><img src='data:image/jpeg;base64,{b64}'>"
                 f"{overlay_svg(d0, w, hh, s, models[0]['thr'], crop)}</div>"
                 f"<div class='meta'>{html.escape(Path(p).name)} · {W}×{H} px · {d0['n_tiles']} {'cửa sổ' if d0['mode'] == 'full' else 'mảnh'} "
                 f"({'cửa sổ trượt trên đặc trưng toàn ảnh, gộp bằng chú ý' if d0['mode'] == 'full' else 'gộp bằng chú ý (MIL)' if d0['mil'] else 'gộp trung bình'})</div></div><div class='res'><h3>Kết quả</h3>")
        for m in models:
            d, thr = m["det"][i], m["thr"]
            ai = d["prob"] >= thr
            col = "var(--ai)" if ai else "var(--real)"
            h.append(f"<div class='row'><div class='top'><span class='nm'>{html.escape(m['name'])}</span>"
                     f"<span class='chip {'ai' if ai else 'real'}'>{'Dự đoán AI' if ai else 'Dự đoán ảnh thật'}</span></div>"
                     f"<div class='sc'>{d['prob']:.4f}</div><div class='bar'><i style='width:{d['prob'] * 100:.1f}%;"
                     f"background:{col}'></i><b style='left:{thr * 100:.1f}%'></b></div>"
                     f"<div class='sub'><span>Điểm AI · ngưỡng VAL {thr:.4f}</span><span>{d['n_tiles']} {'cửa sổ' if d['mode'] == 'full' else 'mảnh'}</span></div></div>")
        h.append("</div></div>")
    h.append("</div><footer>Điểm AI là đầu ra sigmoid của mô hình, chưa phải xác suất đã hiệu chuẩn. Điểm ≥ ngưỡng → dự đoán AI. "
             "Ngưỡng chọn trên tập VAL (chỉ số Youden), không dùng tập test. Màu từng mảnh là điểm riêng của mảnh đó, "
             "chỉ để quan sát; quyết định cuối dựa trên điểm của cả ảnh.</footer></body></html>")
    Path(out).write_text("\n".join(h), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=[], help="file ảnh hoặc thư mục (đệ quy)")
    ap.add_argument("--runs", nargs="*", default=None, help="các run cần so sánh (run đầu là mô hình chính)")
    ap.add_argument("--run", help="(tương thích bản cũ) 1 run")
    ap.add_argument("--html", help="ghi trang HTML kết quả")
    ap.add_argument("--csv", help="ghi bảng kết quả CSV")
    ap.add_argument("--mode", choices=["center", "tiles"], default="tiles",
                    help="mô hình cũ: tiles = toàn ảnh (mặc định) | center = 1 vùng giữa. Mô hình MIL luôn dùng toàn ảnh")
    ap.add_argument("--max-tiles", type=int, default=None, help="giới hạn số mảnh mỗi ảnh (0 = phủ kín toàn ảnh)")
    ap.add_argument("--list-runs", action="store_true")
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    cfg["eval"]["mode"] = a.mode
    if a.max_tiles is not None:
        cfg["eval"]["max_tiles"] = a.max_tiles
    runs, have = pick_runs(cfg, a.runs or ([a.run] if a.run else None))
    if a.list_runs:
        print("\n".join(have)); return
    paths = collect_images(a.images)
    if not paths:
        raise SystemExit("Chưa có ảnh: --images anh.jpg thu_muc/")
    models = []
    for run in runs:
        model, rcfg = load_run(cfg, run)
        rcfg["paths"], rcfg["eval"] = cfg["paths"], dict(cfg["eval"], batch_size=min(cfg["eval"]["batch_size"], 16))
        thr = rcfg.get("_val_threshold") or 0.5
        log.info(f"{run}: {nice_name(run)} | ngưỡng {thr:.4f} | {'MIL' if getattr(model, 'is_mil', False) else a.mode}")
        det = predict_detail(model, rcfg, paths)
        models.append(dict(run=run, name=nice_name(run), thr=thr, det=det))
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"\n{'Ảnh':40s} " + " ".join(f"{m['name'][:22]:>24s}" for m in models))
    rows = []
    for i, p in enumerate(paths):
        cells = []
        for m in models:
            d = m["det"][i]; ai = d["prob"] >= m["thr"]
            cells.append(f"{d['prob']:.3f} {'AI' if ai else 'THẬT':>4s} ({d['n_tiles']})")
            rows.append(dict(image=str(p), width=d["size"][0], height=d["size"][1], run=m["run"], model=m["name"],
                             n_tiles=d["n_tiles"], score=round(d["prob"], 6), threshold=round(m["thr"], 6),
                             verdict="ai" if ai else "real", mil=d["mil"]))
        print(f"{str(p)[-40:]:40s} " + " ".join(f"{c:>24s}" for c in cells))
    if a.csv:
        import pandas as pd
        pd.DataFrame(rows).to_csv(a.csv, index=False); log.info(f"Đã ghi {a.csv}")
    if a.html:
        write_html(a.html, models, paths, cfg["eval"]["crop"]); log.info(f"Đã ghi {a.html} — mở bằng trình duyệt")


if __name__ == "__main__":
    main()
