"""Backend demo phát hiện ảnh AI: 4 mô hình ViT đơn đặc trưng, gọi lại pipeline gốc trong AIGI_ROOT/src.

Chạy:  python app.py            (127.0.0.1:8000; đổi bằng biến môi trường HOST / PORT / AIGI_ROOT / MAX_SIDE)
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import struct
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
# Thứ tự: biến môi trường AIGI_ROOT -> "aigi" cạnh backend/ (gói mang đi) -> "extracted/aigi_final" cạnh web_demo/
_BUNDLED = BASE / "aigi"
AIGI_ROOT = Path(os.environ.get("AIGI_ROOT") or (_BUNDLED if (_BUNDLED / "src").is_dir()
                                                  else BASE.parent / "extracted" / "aigi_final")).resolve()
if not (AIGI_ROOT / "src" / "evaluate.py").is_file():
    raise SystemExit(f"Không thấy pipeline gốc ở {AIGI_ROOT} (thiếu src/evaluate.py). "
                     "Đặt biến môi trường AIGI_ROOT trỏ tới thư mục aigi_final.")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # checkpoint đã đủ trọng số: không được tải gì từ mạng
sys.dont_write_bytecode = True  # không ghi __pycache__ vào thư mục source gốc (chỉ đọc)
sys.path.insert(0, str(AIGI_ROOT))

import torch  # noqa: E402
from fastapi import FastAPI, File, Form, Request, UploadFile  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from PIL import Image  # noqa: E402
from starlette.exceptions import HTTPException  # noqa: E402
from src.config import load_config  # noqa: E402
from src.evaluate import load_run, predict_detail  # noqa: E402

log = logging.getLogger("uvicorn.error")
MAX_BYTES = 10 * 1024 * 1024
MAX_BODY = MAX_BYTES + 64 * 1024  # phần dư cho khung multipart; kích thước file được kiểm tra chính xác trong handler
DRAIN_CAP = 64 * 1024 * 1024
MAX_SIDE = int(os.environ.get("MAX_SIDE", "4096"))  # cạnh dài tối đa (px); 2048 ~ 30 giây, 4096 ~ hơn 2 phút trên CPU
if MAX_SIDE < 256:
    raise SystemExit(f"MAX_SIDE={MAX_SIDE} không hợp lệ: phải >= 256 (kích thước một cửa sổ).")
FORMATS = {"JPEG", "MPO", "PNG", "WEBP"}  # MPO = JPEG nhiều khung của một số máy ảnh/điện thoại
MAX_PENDING = 3  # 1 suy luận đang chạy + tối đa 2 request chờ; vượt thì 503 ngay
DECODE_ERRORS = (OSError, SyntaxError, ValueError, EOFError, struct.error)  # các lỗi Pillow ném khi file hỏng
DIST = BASE / "frontend" / "dist"
API_METHODS = {"health": "GET", "models": "GET", "predict": "POST"}
HTTP_MESSAGES = {400: "Yêu cầu không hợp lệ.", 404: "Không tìm thấy đường dẫn.", 405: "Phương thức không được hỗ trợ."}
Image.MAX_IMAGE_PIXELS = MAX_SIDE * MAX_SIDE

SPECS = {
    "rgb": ("VIT1_rgb_deb_full_s0", "ViT + RGB", "Giá trị màu gốc (đường cơ sở)"),
    "npr": ("VIT2_npr_deb_full_s0", "ViT + NPR", "Chênh lệch giữa các điểm ảnh lân cận"),
    "hh": ("VIT2_hh_deb_full_s0", "ViT + HH", "Băng chéo wavelet Haar (nhiễu tần số cao)"),
    "wpt": ("VIT2_wpt_deb_full_s0", "ViT + WPT", "16 dải tần wavelet packet"),
}
MODELS: dict[str, dict] = {}
# ponytail: hàng chờ trong một tiến trình (1 chạy + 2 chờ); cần worker/hàng đợi riêng nếu phục vụ nhiều người dùng
TURN = asyncio.Lock()
pending = 0


class ApiError(HTTPException):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(status, message)
        self.code = code


def load_models() -> None:
    """Dựng cfg đúng như src.demo.main rồi nạp 4 run. Lỗi nạp -> dừng khởi động, không tạo kết quả giả."""
    cfg = load_config()
    cfg["paths"]["runs"] = str(AIGI_ROOT / "runs")  # config.local.yaml để đường dẫn tương đối theo cwd
    cfg["eval"]["mode"] = "tiles"
    for mid, (run, name, desc) in SPECS.items():
        model, rcfg = load_run(cfg, run)
        rcfg["paths"], rcfg["eval"] = cfg["paths"], dict(cfg["eval"], batch_size=min(cfg["eval"]["batch_size"], 16))
        thr = rcfg.get("_val_threshold")
        if thr is None:
            raise RuntimeError(f"{run}: checkpoint không có ngưỡng VAL (_val_threshold)")
        if getattr(model, "needs_clip", False):
            raise RuntimeError(f"{run}: mô hình cần CLIP, nằm ngoài phạm vi demo")
        MODELS[mid] = dict(id=mid, name=name, description=desc, threshold=float(thr), model=model, rcfg=rcfg)


def check_image(data: bytes) -> None:
    """Xác thực theo nội dung file; kích thước lấy từ header trước khi giải mã (chống decompression bomb)."""
    too_big = ApiError(413, "file_too_large", f"Ảnh quá lớn: cạnh dài tối đa {MAX_SIDE} px.")
    try:
        im = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError:
        raise too_big from None
    except DECODE_ERRORS:
        raise ApiError(400, "invalid_image", "File không phải ảnh hợp lệ.") from None
    if im.format not in FORMATS:
        raise ApiError(415, "unsupported_type", "Chỉ hỗ trợ ảnh JPG, PNG hoặc WEBP.")
    if max(im.size) > MAX_SIDE:
        raise too_big
    if im.mode[0] in "IF":  # I, I;16*, F: pipeline gốc bão hoà thành ảnh gần trắng -> điểm vô nghĩa
        raise ApiError(415, "unsupported_type", "Không hỗ trợ ảnh 16-bit/32-bit mỗi kênh; hãy chuyển sang ảnh 8-bit.")
    try:
        im.load()  # giải mã thật: bắt file bị cắt cụt trước khi vào pipeline
    except DECODE_ERRORS:
        raise ApiError(400, "invalid_image", "Ảnh bị hỏng hoặc không đọc được.") from None


def run_predict(m: dict, data: bytes) -> dict:
    """Chạy trong threadpool, chỉ sau khi request đã tới lượt (TURN) -> mỗi lúc tối đa một file tạm, một suy luận."""
    check_image(data)
    fd, path = tempfile.mkstemp(suffix=".img")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)  # ghi nguyên byte gốc: không lưu lại qua Pillow để kết quả trùng pipeline gốc
        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                d = predict_detail(m["model"], m["rcfg"], [path])[0]
        except Exception:
            log.exception("Suy luận thất bại (model_id=%s)", m["id"])
            raise ApiError(500, "inference_failed", "Suy luận thất bại.") from None
        ms = (time.perf_counter() - t0) * 1000
    finally:
        try:
            os.unlink(path)
        except OSError as e:  # Windows: file còn handle mở thì không xoá được -> để sót file tạm còn hơn làm hỏng kết quả
            log.warning("Không xoá được file tạm sau khi suy luận: %s", e)
    score = float(d["prob"])
    return dict(model_id=m["id"], model_name=m["name"], label="ai" if score >= m["threshold"] else "real",
                score_ai=score, score_real=1.0 - score, threshold=m["threshold"], inference_ms=round(ms, 1),
                image=dict(width=d["size"][0], height=d["size"][1]), n_windows=int(d["n_tiles"]))


class BodyLimit:
    """Chặn body > MAX_BODY ngay khi đọc, kể cả khi client gửi chunked / Content-Length sai."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/api/predict":
            return await self.app(scope, receive, send)
        seen = 0

        async def limited():
            nonlocal seen
            msg = await receive()
            seen += len(msg.get("body", b""))
            if seen > MAX_BODY:
                # đọc bỏ phần còn lại (có trần) để client nhận được 413 thay vì đứt kết nối giữa chừng
                while msg.get("more_body") and seen < DRAIN_CAP:
                    msg = await receive()
                    seen += len(msg.get("body", b""))
                raise ApiError(413, "file_too_large", "File vượt quá 10 MB.")
            return msg

        await self.app(scope, limited, send)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_models()
    yield


app = FastAPI(title="AIGI demo", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(BodyLimit)


def _err(status: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=headers)


@app.exception_handler(HTTPException)
async def on_http_error(_, e: HTTPException):
    if isinstance(e, ApiError):
        return _err(e.status_code, e.code, str(e.detail))
    return _err(e.status_code, "http_error", HTTP_MESSAGES.get(e.status_code, "Yêu cầu không thực hiện được."), e.headers)


@app.exception_handler(Exception)
async def on_unexpected_error(_, __: Exception):  # traceback do ServerErrorMiddleware/uvicorn ghi log
    return _err(500, "inference_failed", "Lỗi máy chủ khi xử lý ảnh.")


@app.exception_handler(RequestValidationError)
async def on_validation_error(_, e: RequestValidationError):
    if any("model_id" in err["loc"] for err in e.errors()):
        return _err(422, "unknown_model", "Thiếu model_id.")
    return _err(400, "invalid_image", "Thiếu file ảnh.")


@app.get("/api/health")
async def health():
    return {"ok": True, "models_loaded": list(MODELS), "max_side": MAX_SIDE}


@app.get("/api/models")
async def models():
    return [{k: m[k] for k in ("id", "name", "description", "threshold")} for m in MODELS.values()]


@app.post("/api/predict")
async def predict(request: Request, file: UploadFile = File(...), model_id: str = Form(...)):
    global pending
    m = MODELS.get(model_id)
    if m is None:
        raise ApiError(422, "unknown_model", f"model_id phải là một trong: {', '.join(MODELS)}.")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ApiError(413, "file_too_large", "File vượt quá 10 MB.")
    if pending >= MAX_PENDING:
        raise ApiError(503, "busy", "Máy chủ đang bận (1 ảnh đang chấm, 2 ảnh đang chờ). Hãy thử lại sau.")
    pending += 1
    try:
        async with TURN:  # chờ trên event loop, không giữ luồng nào của threadpool
            if await request.is_disconnected():
                log.info("Bỏ request: client đã ngắt kết nối trong lúc chờ lượt")
                raise ApiError(499, "client_closed", "Client đã ngắt kết nối.")
            return await run_in_threadpool(run_predict, m, data)
    finally:
        pending -= 1


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def api_fallback(path: str):
    """Đứng trước mount tĩnh: đường dẫn API lạ -> 404, sai phương thức -> 405, đều theo dạng {"error": ...}."""
    if path in API_METHODS:
        raise HTTPException(405, headers={"Allow": API_METHODS[path]})
    raise HTTPException(404)


if DIST.is_dir():  # bản build của frontend (nếu có) được phục vụ cùng cổng
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
