# Web demo local: phát hiện ảnh do AI tạo (4 mô hình ViT đơn đặc trưng)

Tải ảnh lên, chọn 1 trong 4 mô hình (RGB, NPR, HH, WPT), nhận nhãn AI-Generated / Real Image.
Chỉ chạy local, suy luận CPU. Backend gọi lại nguyên pipeline gốc trong `aigi/src` (không viết lại
tiền xử lý, trích đặc trưng hay kiến trúc).

```
web_demo/
  backend/app.py              FastAPI: API + phục vụ frontend/dist (nếu có)
  backend/requirements.txt       phụ thuộc trực tiếp cho suy luận (đã ghim phiên bản)
  backend/requirements.lock.txt  khoá đầy đủ, kể cả phụ thuộc gián tiếp (nên cài từ file này)
  backend/requirements-dev.txt   thêm httpx, pandas cho kiểm thử
  backend/tests/check_all.py  kiểm thử đầu-cuối qua HTTP
  frontend/                   Vite + React (build ra frontend/dist)
  Dockerfile, BAO_CAO.md
```

## Chạy trên máy mới (từng bước)

Đã kiểm chứng trên Linux / WSL2 (Ubuntu, x86_64). Phần Windows ở cuối mục chưa được chạy thử.

### Bước 0. Chuẩn bị

| Cần có | Ghi chú |
|---|---|
| Git | để tải repo |
| Python 3.12 | hoặc cài `uv` (https://docs.astral.sh/uv/), uv tự tải Python 3.12 |
| Node.js 20 trở lên | chỉ để build giao diện một lần (máy thử dùng Node 24) |
| Khoảng 2,5 GB ổ đĩa, 2 GB RAM trống, có Internet ở lần cài đầu | không cần GPU |

### Bước 1. Tải repo

```bash
git clone https://github.com/HoaNguyen55/aigi-web-demo.git
cd aigi-web-demo
```

### Bước 2. Source + checkpoint gốc (đã có sẵn trong repo)

Repo đã kèm thư mục `aigi/` (khoảng 334 MB) nên không cần `aigi_final.zip`:

```
aigi/src/                                 code pipeline gốc (nguyên trạng)
aigi/config.yaml, aigi/config.local.yaml
aigi/runs/VIT1_rgb_deb_full_s0/final.pt   ViT + RGB
aigi/runs/VIT2_npr_deb_full_s0/final.pt   ViT + NPR
aigi/runs/VIT2_hh_deb_full_s0/final.pt    ViT + HH
aigi/runs/VIT2_wpt_deb_full_s0/final.pt   ViT + WPT
aigi/demo_anh/                            8 ảnh mẫu
```

`git clone` vì vậy tải khoảng 330 MB. Muốn dùng bản source/checkpoint khác thì đặt biến môi trường `AIGI_ROOT`
trỏ tới thư mục đó khi chạy (bước 5).

### Bước 3. Cài thư viện Python (PyTorch bản CPU)

```bash
# Cách 1: uv (cách đã kiểm chứng; venv tạo bằng uv KHÔNG có pip)
uv venv .venv --python 3.12
uv pip install -p .venv -r backend/requirements.lock.txt

# Cách 2: venv + pip chuẩn (chưa chạy thử)
python3.12 -m venv .venv
.venv/bin/pip install -r backend/requirements.lock.txt
```

Bước này tải khoảng vài trăm MB. Trên WSL, nếu repo nằm ở ổ Windows (`/mnt/c`, `/mnt/d`) thì nên tạo venv ở thư mục
home (ví dụ `~/aigi_web_venv`) vì ổ Windows rất chậm; khi đó thay `.venv` bằng đường dẫn đó ở các lệnh bên dưới.

### Bước 4. Build giao diện (một lần)

```bash
cd frontend
npm install
npm run build        # tạo frontend/dist
cd ..
```

### Bước 5. Chạy server

Từ lần sau chỉ cần bước này: bước 3 và 4 làm một lần (làm lại bước 4 khi giao diện thay đổi, bước 3 khi file requirements thay đổi).

```bash
cd backend
../.venv/bin/python app.py
```

Chờ khoảng 20 giây cho tới khi thấy dòng `Uvicorn running on http://127.0.0.1:8000`, rồi mở trình duyệt tại
**http://127.0.0.1:8000**. Dừng server bằng Ctrl+C.

Các biến thể:

```bash
PORT=8001 ../.venv/bin/python app.py                           # đổi cổng khi 8000 đang bận
AIGI_ROOT=/duong/dan/aigi_final ../.venv/bin/python app.py     # source gốc nằm ở chỗ khác
MAX_SIDE=2048 ../.venv/bin/python app.py                       # chặn ảnh có cạnh dài trên 2048 px
```

### Bước 6. Kiểm tra nhanh

```bash
curl http://127.0.0.1:8000/api/health
# {"ok":true,"models_loaded":["rgb","npr","hh","wpt"],"max_side":4096}
```

Trên trang web: tải một ảnh trong `aigi/demo_anh/`, chọn mô hình, bấm "Analyze Image".

### Lỗi thường gặp

| Hiện tượng | Nguyên nhân và cách xử lý |
|---|---|
| `Không thấy pipeline gốc ở ... (thiếu src/evaluate.py)` | Thư mục `aigi/` bị thiếu hoặc `AIGI_ROOT` trỏ sai. Kiểm tra `aigi/src/evaluate.py` tồn tại. |
| `FileNotFoundError ... final.pt` khi khởi động | Thiếu một trong 4 checkpoint ở `aigi/runs/<run>/final.pt` (clone chưa xong hoặc file bị xoá). |
| Mở http://127.0.0.1:8000 báo 404 nhưng `/api/health` vẫn chạy | Chưa build giao diện. Làm bước 4 rồi khởi động lại server. |
| Trang báo "Không tải được danh sách mô hình" | Server đang chạy bản cũ hơn giao diện. Khởi động lại server. |
| `address already in use` | Cổng 8000 đang bận. Chạy với `PORT=8001`. |
| Ảnh lớn chạy rất lâu | Suy luận CPU: ảnh 1024 px khoảng 7–10 giây, ảnh điện thoại 1920 × 2560 px khoảng 30 giây mỗi mô hình. |

### Windows (chưa chạy thử)

Các bước giống hệt, chỉ khác lệnh. File khoá `requirements.lock.txt` được tạo trên Linux nên trên Windows dùng `requirements.txt`:

```bat
uv venv .venv --python 3.12
uv pip install -p .venv -r backend\requirements.txt
cd frontend && npm install && npm run build && cd ..
cd backend
set PYTHONUTF8=1
..\.venv\Scripts\python app.py
```

Nếu gặp lỗi ở bước cài PyTorch hoặc khi chạy, dùng WSL2 và làm theo các bước Linux ở trên.

## Ghi chú cài đặt

`requirements.lock.txt` ghim mọi gói (Python 3.12, Linux x86_64) nên cài lại ra đúng môi trường đã thử.
`requirements.txt` chỉ ghim phụ thuộc trực tiếp; dùng nó khi muốn nâng phiên bản, rồi tạo lại file khoá bằng
`uv pip freeze` trên một venv sạch. Nếu chưa có `frontend/dist`, API vẫn chạy bình thường (chỉ không có trang ở `/`).

Biến môi trường:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `AIGI_ROOT` | `aigi/` cạnh `backend/`, nếu không có thì `../extracted/aigi_final` | thư mục source + `runs/` gốc (chỉ đọc) |
| `MAX_SIDE` | `4096` | cạnh dài tối đa của ảnh (px); ảnh lớn hơn bị từ chối 413. Đặt `2048` để mỗi ảnh không quá khoảng 35 giây |
| `HOST` | `127.0.0.1` | địa chỉ bind |
| `PORT` | `8000` | cổng |

## Tuỳ chọn trên giao diện

- **Giới hạn cạnh ảnh:** chọn 512 / 1024 / ... tới trần `MAX_SIDE` của máy chủ. Ảnh vượt mức bị chặn ngay trên trình duyệt,
  không gửi lên máy chủ. Không làm đổi điểm AI.
- **Ngưỡng quyết định:** để trống = ngưỡng của checkpoint. Nhập giá trị 0–1 để đổi ngưỡng; chỉ nhãn AI/Real đổi theo,
  điểm AI giữ nguyên và không cần chạy lại.
- **Chạy cả 4 mô hình và so sánh:** gọi `/api/predict` lần lượt cho 4 mô hình, hiện bảng so sánh.

`GET /api/health` trả thêm `max_side` để giao diện biết trần của máy chủ.

## API

- `GET /api/health` → `{ "ok": true, "models_loaded": [...] }`
- `GET /api/models` → `[{ id, name, description, threshold }]`
- `POST /api/predict` (multipart: `file`, `model_id`) →
  `{ model_id, model_name, label, score_ai, score_real, threshold, inference_ms, image: {width, height}, n_windows }`
- Lỗi: `{ "error": { "code", "message" } }`

| Tình huống | HTTP | code |
|---|---|---|
| Không phải ảnh / ảnh hỏng / thiếu `file` | 400 | `invalid_image` |
| Ảnh hợp lệ nhưng không phải JPEG, PNG, WEBP; hoặc ảnh 16-bit/32-bit mỗi kênh | 415 | `unsupported_type` |
| File > 10 MB, hoặc cạnh dài > `MAX_SIDE` (mặc định 4096 px) | 413 | `file_too_large` |
| `model_id` lạ hoặc thiếu | 422 | `unknown_model` |
| Lỗi khi suy luận hoặc lỗi máy chủ khác (có traceback trong log server) | 500 | `inference_failed` |
| Quá tải: đã có 1 ảnh đang chấm + 2 ảnh đang chờ | 503 | `busy` |
| Đường dẫn `/api/...` không tồn tại / sai phương thức / body multipart hỏng | 404 / 405 / 400 | `http_error` |

### Hàng chờ và thời gian

Mỗi lúc chỉ chạy một suy luận. Server nhận tối đa 3 request `/api/predict` cùng lúc (1 đang chạy + 2 chờ, phục vụ
theo thứ tự đến); request thứ 4 trở đi nhận 503 `busy` ngay lập tức, không phải chờ. Request đã được nhận thì chờ tới
lượt, không có timeout phía server; nếu client ngắt kết nối trong lúc chờ thì request đó bị bỏ qua khi tới lượt.
Suy luận đã bắt đầu thì không huỷ được.

Thời gian ước tính trên CPU 4 nhân (khoảng 150 ms mỗi cửa sổ 256×256, bước 128 px; số đo trong `BAO_CAO.md`):

| Cạnh ảnh | Số cửa sổ | Thời gian một ảnh |
|---|---|---|
| ≤ 256 px | 1 | 0,1–0,6 giây |
| 512 px | 9 | 1–1,5 giây |
| 1024 px | 49 | 7–10 giây |
| 2048 px | 225 | khoảng 35 giây |
| 4096 px (tối đa mặc định) | 961 | khoảng 2,5 phút |

Trường hợp xấu nhất một request phải chờ thêm 2 ảnh phía trước, nên client cần timeout dài (frontend đặt 5 phút;
ba ảnh 4096 px liên tiếp vẫn có thể vượt mức đó).

### Lưu ý về ảnh đầu vào

Đây là hành vi của pipeline gốc (`Image.open(path).convert("RGB")`), backend giữ nguyên để kết quả không lệch:

- Kênh alpha bị bỏ (không trộn với nền).
- EXIF orientation không được áp dụng: ảnh được chấm theo đúng ma trận điểm ảnh lưu trong file, `image.width/height`
  là kích thước đó.
- Ảnh động (WEBP/APNG) chỉ chấm khung đầu tiên.
- Ảnh không bị thu nhỏ; ảnh nhỏ hơn 256 px được đệm phản chiếu.
- Ảnh 16-bit/32-bit mỗi kênh (mode `I`, `I;16`, `F`) bị từ chối với 415: pipeline gốc sẽ bão hoà chúng thành ảnh gần
  trắng nên điểm vô nghĩa. Hãy chuyển sang 8-bit trước.
- File Pillow nhận dạng là `MPO` (JPEG nhiều khung của một số điện thoại) được chấp nhận như JPEG.

```bash
curl -F file=@anh.jpg -F model_id=rgb http://127.0.0.1:8000/api/predict
```

`score_ai` là đầu ra sigmoid của mô hình, **chưa hiệu chuẩn**, không phải accuracy hay xác suất thật.

## Kiểm thử

```bash
uv pip install -p ~/aigi_web_venv -r backend/requirements-dev.txt     # hoặc ~/aigi_web_venv/bin/pip install -r ...
~/aigi_web_venv/bin/python backend/tests/check_all.py        # tự bật server ở một cổng rảnh rồi tắt
```

Script kiểm tra: nạp 4 checkpoint + ngưỡng + tên mô hình; so P(AI) của `/api/predict` trên 8 ảnh `demo_anh` × 4 mô
hình với pipeline gốc theo hai cách (`python -m src.demo --csv`, và gọi thẳng `load_run` + `predict_detail` không làm
tròn); các ca lỗi; quá tải 8 request đồng thời; thời gian suy luận 256/512/1024 px; thư mục tạm của server không còn
file. Mất khoảng 5 phút (`--skip-timing` để bỏ phần đo thời gian). Kết quả đã chạy: xem `BAO_CAO.md`.

## Cập nhật checkpoint

- Ghi đè `final.pt` trong `AIGI_ROOT/runs/<run>/` rồi khởi động lại server. Ngưỡng được đọc lại từ checkpoint
  (`val_threshold`), không có giá trị nào hardcode trong backend.
- Đổi sang run khác (tên thư mục khác): sửa bảng `SPECS` trong `backend/app.py`.
- Dùng bản source/checkpoint ở nơi khác: đặt `AIGI_ROOT`.
- `final.pt` được nạp bằng `torch.load(weights_only=False)` (pickle): chỉ dùng checkpoint do chính bạn tạo.
- Sau khi đổi checkpoint, chạy lại `check_all.py` (bảng ngưỡng đối chiếu trong script cần cập nhật theo).

## Docker (CPU)

```bash
docker build -t aigi-web .
docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/aigi":/aigi:ro aigi-web
```

Image không chứa source/checkpoint gốc; thư mục `aigi/` của repo được gắn chỉ-đọc vào `/aigi`. Có `HEALTHCHECK` gọi
`/api/health`. Dockerfile **chưa được build thử** (máy phát triển không có docker).
