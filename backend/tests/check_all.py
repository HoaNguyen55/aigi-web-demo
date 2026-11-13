"""Kiểm thử đầu-cuối qua HTTP thật: tự bật server ở cổng thử, chạy 5 nhóm kiểm thử, rồi tắt server.

  ~/aigi_web_venv/bin/python backend/tests/check_all.py            # tất cả
  ~/aigi_web_venv/bin/python backend/tests/check_all.py --skip-timing

Cần requirements-dev.txt (httpx, pandas). Thoát mã 1 nếu có kiểm thử không đạt.
"""
from __future__ import annotations

import argparse
import csv
import json
import io
import os
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx
from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
AIGI_ROOT = Path(os.environ.get("AIGI_ROOT", "/mnt/d/xulyanh/extracted/aigi_final"))
DEMO = AIGI_ROOT / "demo_anh"
EXPECTED = {"rgb": 0.8870, "npr": 0.7236, "hh": 0.6874, "wpt": 0.5413}
RUNS = {"rgb": "VIT1_rgb_deb_full_s0", "npr": "VIT2_npr_deb_full_s0", "hh": "VIT2_hh_deb_full_s0",
        "wpt": "VIT2_wpt_deb_full_s0"}
TIMING_IMAGES = {256: "AI_adm_584_adm_153.png", 512: "AI_sd14_025_sdv4_00020.png", 1024: "AI_mj_768_midjourney_197.png"}
NAMES = {"rgb": "ViT + RGB", "npr": "ViT + NPR", "hh": "ViT + HH", "wpt": "ViT + WPT"}
DIST = BACKEND.parent / "frontend" / "dist"
FAILS: list[str] = []
# Đối chiếu trực tiếp: load_run + predict_detail của pipeline gốc trong tiến trình riêng, điểm KHÔNG làm tròn
DIRECT = """
import json, sys
from pathlib import Path
from src.config import load_config
from src.evaluate import load_run, predict_detail
cfg = load_config(); cfg["eval"]["mode"] = "tiles"
paths = sorted(Path("demo_anh").iterdir())
out = {}
for run in sys.argv[1:]:
    model, rcfg = load_run(cfg, run)
    rcfg["paths"], rcfg["eval"] = cfg["paths"], dict(cfg["eval"], batch_size=min(cfg["eval"]["batch_size"], 16))
    for d in predict_detail(model, rcfg, paths):
        out[run + "|" + Path(d["path"]).name] = [d["prob"], d["size"][0], d["size"][1], d["n_tiles"]]
print("JSON" + json.dumps(out))
"""


def check(ok: bool, what: str) -> None:
    print(f"  [{'OK' if ok else 'FAIL'}] {what}")
    if not ok:
        FAILS.append(what)


def img_bytes(fmt: str, size=(64, 64), mode: str = "RGB") -> bytes:
    b = io.BytesIO()
    Image.new(mode, size, (120, 30, 200) if mode == "RGB" else 40000).save(b, fmt)
    return b.getvalue()


def post(c: httpx.Client, data: bytes, model_id: str | None = "rgb", name: str = "a.jpg") -> httpx.Response:
    return c.post("/api/predict", files={"file": (name, data)}, data={"model_id": model_id} if model_id else {})


def test_thresholds(c: httpx.Client) -> None:
    print("\n1. Nạp 4 checkpoint, ngưỡng đọc từ checkpoint")
    h = c.get("/api/health").json()
    check(h["ok"] and sorted(h["models_loaded"]) == sorted(EXPECTED), f"/api/health: {h}")
    got = {m["id"]: m["threshold"] for m in c.get("/api/models").json()}
    for mid, exp in EXPECTED.items():
        check(abs(got.get(mid, -1) - exp) < 5e-5, f"{mid}: ngưỡng đọc {got.get(mid)!r} | bảng {exp:.4f}")
    names = {m["id"]: m["name"] for m in c.get("/api/models").json()}
    check(names == NAMES, f"tên mô hình: {names}")


def test_consistency(c: httpx.Client, py: str) -> None:
    print("\n2. Nhất quán với pipeline gốc (tiến trình riêng) trên 8 ảnh demo_anh x 4 mô hình")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
    api = {(mid, p.name): post(c, p.read_bytes(), mid, p.name).json() for mid in RUNS for p in sorted(DEMO.iterdir())}
    check(all(j["model_name"] == NAMES[mid] for (mid, _), j in api.items()), "model_name của /api/predict đúng tên mô hình")

    print("  2a. python -m src.demo --csv (điểm trong CSV làm tròn 6 chữ số)")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "goc.csv"
        subprocess.run([py, "-m", "src.demo", "--runs", *RUNS.values(), "--images", "demo_anh", "--csv", str(out)],
                       cwd=AIGI_ROOT, check=True, stdout=subprocess.DEVNULL, env=env)
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    check(len(rows) == 32, f"src.demo trả {len(rows)} dòng (mong đợi 8 ảnh x 4 mô hình = 32)")
    worst = 0.0
    for r in rows:
        mid, name = next(k for k, v in RUNS.items() if v == r["run"]), Path(r["image"]).name
        j = api[mid, name]
        worst = max(worst, abs(j["score_ai"] - float(r["score"])))
        if (j["label"], j["n_windows"], j["image"]["width"], j["image"]["height"]) != (
                r["verdict"], int(r["n_tiles"]), int(r["width"]), int(r["height"])):
            check(False, f"{mid}/{name}: nhãn, số cửa sổ hoặc kích thước khác src.demo")
    check(worst <= 1e-5, f"src.demo: sai lệch tuyệt đối lớn nhất trên {len(rows)} cặp: {worst:.2e} (mục tiêu <= 1e-5)")

    print("  2b. load_run + predict_detail trực tiếp (điểm không làm tròn)")
    res = subprocess.run([py, "-c", DIRECT, *RUNS.values()], cwd=AIGI_ROOT, check=True, capture_output=True, text=True, env=env)
    ref = json.loads(next(l for l in res.stdout.splitlines() if l.startswith("JSON"))[4:])
    check(len(ref) == 32, f"đối chiếu trực tiếp trả {len(ref)} kết quả (mong đợi 32)")
    worst = 0.0
    for key, (prob, w, h, n) in sorted(ref.items()):
        run, name = key.split("|")
        mid = next(k for k, v in RUNS.items() if v == run)
        j = api[mid, name]
        d = abs(j["score_ai"] - prob)
        worst = max(worst, d)
        print(f"    {mid:4s} {name[:34]:34s} api {j['score_ai']!r} | gốc {prob!r} | lệch {d:.1e} | {j['label']}")
        if (j["n_windows"], j["image"]["width"], j["image"]["height"]) != (n, w, h):
            check(False, f"{mid}/{name}: số cửa sổ hoặc kích thước khác predict_detail")
    check(worst <= 1e-5, f"trực tiếp: sai lệch tuyệt đối lớn nhất trên {len(ref)} cặp: {worst:.2e} (mục tiêu <= 1e-5)")


def test_errors(c: httpx.Client) -> None:
    print("\n3. Ca lỗi")
    good = (DEMO / TIMING_IMAGES[256]).read_bytes()
    big = os.urandom(11 * 1024 * 1024)
    boundary = "kiemthu"
    head = (f'--{boundary}\r\nContent-Disposition: form-data; name="model_id"\r\n\r\nrgb\r\n--{boundary}\r\n'
            f'Content-Disposition: form-data; name="file"; filename="a.jpg"\r\n\r\n').encode()

    def chunked():
        yield head
        for i in range(0, len(big), 65536):
            yield big[i:i + 65536]
        yield f"\r\n--{boundary}--\r\n".encode()

    webp = io.BytesIO()
    Image.open(DEMO / TIMING_IMAGES[256]).save(webp, "WEBP", lossless=True)
    r = post(c, webp.getvalue(), "rgb", "a.webp")
    check(r.status_code == 200 and r.json()["image"] == {"width": 256, "height": 256}, f"WEBP hợp lệ: HTTP {r.status_code}")
    cases = [
        ("PNG 16-bit (mode I;16)", lambda: post(c, img_bytes("PNG", mode="I;16"), name="a.png"), 415, "unsupported_type"),
        ("file text đổi đuôi .jpg", lambda: post(c, b"day khong phai anh\n" * 50), 400, "invalid_image"),
        ("JPEG bị cắt cụt", lambda: post(c, (DEMO / "THAT_adm_ILSVRC2012_val_00014053.jpg").read_bytes()[:20000]), 400, "invalid_image"),
        ("GIF đổi đuôi .jpg", lambda: post(c, img_bytes("GIF")), 415, "unsupported_type"),
        ("BMP đổi đuôi .png", lambda: post(c, img_bytes("BMP"), name="a.png"), 415, "unsupported_type"),
        ("file rác đúng 10 MB (không vượt giới hạn)", lambda: post(c, big[:10 * 1024 * 1024]), 400, "invalid_image"),
        ("file 10 MB + 1 byte", lambda: post(c, big[:10 * 1024 * 1024 + 1]), 413, "file_too_large"),
        ("file 11 MB (có Content-Length)", lambda: post(c, big), 413, "file_too_large"),
        ("file 11 MB gửi chunked, không Content-Length",
         lambda: c.post("/api/predict", content=chunked(), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}),
         413, "file_too_large"),
        ("PNG 2049x8 px (cạnh dài > MAX_SIDE=2048)", lambda: post(c, img_bytes("PNG", (2049, 8))), 413, "file_too_large"),
        ("model_id lạ", lambda: post(c, good, "clip"), 422, "unknown_model"),
        ("thiếu model_id", lambda: post(c, good, None), 422, "unknown_model"),
        ("thiếu file", lambda: c.post("/api/predict", data={"model_id": "rgb"}), 400, "invalid_image"),
        ("đường dẫn API không tồn tại", lambda: c.get("/api/khongco"), 404, "http_error"),
        ("sai phương thức (GET /api/predict)", lambda: c.get("/api/predict"), 405, "http_error"),
    ]
    for what, call, status, code in cases:
        r = call()
        try:
            got = r.json()["error"]["code"]
        except Exception:
            got = r.text[:80]
        check(r.status_code == status and got == code, f"{what}: HTTP {r.status_code} {got} (mong đợi {status} {code})")
    r = post(c, img_bytes("PNG", (2048, 8)))
    check(r.status_code == 200, f"PNG 2048x8 px (đúng bằng MAX_SIDE=2048) được nhận: HTTP {r.status_code}")
    check(post(c, good).status_code == 200, "server vẫn phục vụ bình thường sau các ca lỗi")


def test_overload(c: httpx.Client, n: int = 8) -> None:
    print(f"\n4. Quá tải: {n} request đồng thời (ảnh 512 px) -> 1 chạy + 2 chờ được nhận, còn lại 503 ngay")
    data, out = (DEMO / TIMING_IMAGES[512]).read_bytes(), []

    def one():
        t0 = time.perf_counter()
        r = post(c, data, "rgb")
        out.append((r.status_code, r.json().get("error", {}).get("code"), time.perf_counter() - t0))

    ts = [threading.Thread(target=one) for _ in range(n)]
    for t in ts:
        t.start()
    time.sleep(0.5)
    page = "/" if DIST.is_dir() else "/api/health"
    t0 = time.perf_counter()
    ok = c.get(page).status_code == 200
    dt, busy = time.perf_counter() - t0, any(t.is_alive() for t in ts)
    for t in ts:
        t.join()
    check(ok and busy and dt < 1.0, f"GET {page} trả 200 sau {dt * 1000:.0f} ms trong lúc quá tải")
    rejected = [o for o in out if o[0] == 503]
    check(sum(o[0] == 200 for o in out) == 3 and len(rejected) == n - 3,
          f"{sum(o[0] == 200 for o in out)} request trả 200, {len(rejected)} trả 503 (mong đợi 3 và {n - 3})")
    check(all(o[1] == "busy" and o[2] < 2.0 for o in rejected),
          f"503 busy chậm nhất sau {max((o[2] for o in rejected), default=0):.2f} giây (mong đợi < 2)")
    print(f"  các request 200 xong sau: {', '.join(f'{o[2]:.1f}' for o in sorted(out, key=lambda o: o[2]) if o[0] == 200)} giây")
    check(post(c, data, "rgb").status_code == 200, "sau đợt quá tải server nhận request mới bình thường")


def test_timing(c: httpx.Client, pid: int, reps: int) -> None:
    print(f"\n5. Thời gian suy luận CPU (inference_ms do server đo; 1 lần làm nóng + trung vị {reps} lần)")
    rss = next(int(l.split()[1]) for l in Path(f"/proc/{pid}/status").read_text().splitlines() if l.startswith("VmRSS"))
    print(f"  RAM tiến trình server với 4 mô hình đã nạp (sau nhóm 1-3, trước phép đo): {rss / 1024:.0f} MB")
    print("  | mô hình | " + " | ".join(f"{s} px ({TIMING_IMAGES[s]})" for s in TIMING_IMAGES) + " |")
    for mid in RUNS:
        cells = []
        for side, name in TIMING_IMAGES.items():
            data = (DEMO / name).read_bytes()
            js = [post(c, data, mid, name).json() for _ in range(reps + 1)][1:]
            assert js[0]["image"] == {"width": side, "height": side}
            cells.append(f"{statistics.median(j['inference_ms'] for j in js):.0f} ms ({js[0]['n_windows']} cửa sổ)")
        print(f"  | {mid} | " + " | ".join(cells) + " |")
    # /api/health phải trả lời trong lúc đang suy luận
    t = threading.Thread(target=post, args=(c, (DEMO / TIMING_IMAGES[1024]).read_bytes(), "rgb"))
    t.start()
    time.sleep(0.3)
    t0 = time.perf_counter()
    ok = c.get("/api/health").status_code == 200
    dt = time.perf_counter() - t0
    busy = t.is_alive()
    t.join()
    check(ok and busy and dt < 1.0, f"/api/health trả lời sau {dt * 1000:.0f} ms trong lúc đang suy luận ảnh 1024 px")
    rss = next(int(l.split()[1]) for l in Path(f"/proc/{pid}/status").read_text().splitlines() if l.startswith("VmRSS"))
    print(f"  RAM tiến trình server sau phép đo: {rss / 1024:.0f} MB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0, help="0 = tự chọn một cổng đang rảnh")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip-timing", action="store_true")
    a = ap.parse_args()
    py = sys.executable
    if not a.port:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            a.port = s.getsockname()[1]
    tmp = tempfile.mkdtemp(prefix="aigi_srv_tmp_")  # thư mục tạm riêng của server: phải rỗng sau khi chạy
    srv = subprocess.Popen([py, "app.py"], cwd=BACKEND, env=dict(os.environ, PORT=str(a.port), HOST="127.0.0.1", TMPDIR=tmp, MAX_SIDE="2048"))
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{a.port}", timeout=300) as c:
            t0 = time.perf_counter()
            for _ in range(240):
                if srv.poll() is not None:
                    print("Server thoát khi khởi động (xem lỗi phía trên).")
                    return 1
                try:
                    c.get("/api/health")
                    break
                except httpx.TransportError:
                    time.sleep(0.5)
            else:
                print("Server không sẵn sàng sau 120 giây.")
                return 1
            print(f"Server sẵn sàng sau {time.perf_counter() - t0:.1f} giây (khởi động + nạp 4 mô hình)")
            test_thresholds(c)
            test_consistency(c, py)
            test_errors(c)
            test_overload(c)
            if not a.skip_timing:
                test_timing(c, srv.pid, a.reps)
            # torch tự tạo thư mục rỗng torchinductor_<user> khi import -> chỉ đếm file
            left = [f for _, _, fs in os.walk(tmp) for f in fs]
            check(not left, f"thư mục tạm của server không còn file nào sau khi chạy (còn {len(left)} file)")
    finally:
        srv.terminate()
        try:
            srv.wait(20)
        except subprocess.TimeoutExpired:
            srv.kill()
            srv.wait()
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nKẾT QUẢ: {'ĐẠT tất cả' if not FAILS else f'{len(FAILS)} mục KHÔNG ĐẠT: ' + '; '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
