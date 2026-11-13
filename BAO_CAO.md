# Báo cáo: web demo phát hiện ảnh AI (phần backend)

Ngày chạy kiểm thử: 2026-11-13, sau khi sửa theo review độc lập. Máy: WSL2 Ubuntu, 4 nhân CPU, ~20 GB RAM, không GPU.
Python 3.12.3, torch 2.15.0+cpu, torchvision 0.29.2+cpu, timm 1.0.31, FastAPI 0.162.5.
Mọi số dưới đây đo trên bản code hiện tại: một lần chạy trọn vẹn `backend/tests/check_all.py` (đạt tất cả) và các
phép đo tay ghi rõ ở mục 5–6. Số của các lần chạy trước khi sửa đã bị thay thế.

## 1. Kiến trúc

- Một tiến trình FastAPI (`backend/app.py`), bind `127.0.0.1:8000`, phục vụ cả API và bản build tĩnh
  `frontend/dist` (nếu thư mục tồn tại) trên cùng một cổng, không cần CORS.
- Khi khởi động: kiểm tra `AIGI_ROOT/src/evaluate.py` tồn tại (không có → thoát với thông báo rõ), thêm `AIGI_ROOT`
  vào `sys.path`, dựng `cfg` đúng như `src.demo.main` (`load_config()`, `eval.mode = "tiles"`,
  `eval.batch_size = min(64, 16)`, `paths.runs` ghi đè thành đường dẫn tuyệt đối `AIGI_ROOT/runs`), rồi gọi
  `src.evaluate.load_run` cho 4 run. Nạp lỗi hoặc thiếu ngưỡng → server không khởi động.
- Hàng chờ nằm trong handler async, trước threadpool: tối đa 1 suy luận đang chạy + 2 request chờ
  (`asyncio.Lock` + bộ đếm). Request thứ 4 nhận 503 `busy` ngay. Request đang chờ không giữ luồng nào, chưa giải mã
  ảnh và chưa có file tạm. Khi tới lượt, server kiểm tra client còn kết nối không; đã ngắt thì bỏ qua.
- Khi tới lượt (trong threadpool): kiểm tra ảnh → ghi nguyên byte gốc ra file tạm →
  `src.evaluate.predict_detail(model, rcfg, [path])` trong `torch.inference_mode()` → xoá file tạm trong `finally`.
  Không viết lại tiền xử lý, trích đặc trưng hay kiến trúc; không đặt `max_tiles` (dùng mọi cửa sổ).
- Lỗi suy luận được ghi log kèm traceback (`logging.exception`, không có tên file gốc hay nội dung ảnh) rồi trả 500.
- Không ghi gì vào `aigi_final` (tắt ghi `__pycache__`, cache CLIP không dùng); `HF_HUB_OFFLINE=1`, không tải gì từ mạng.

## 2. Checkpoint dùng và ngưỡng

Ngưỡng đọc từ checkpoint lúc khởi động (`rcfg["_val_threshold"]`), không hardcode.

| model_id | name | Run | Ngưỡng đọc được | Bảng đặc tả | Khớp |
|---|---|---|---|---|---|
| rgb | ViT + RGB | runs/VIT1_rgb_deb_full_s0 | 0.886966 | 0.8870 | có |
| npr | ViT + NPR | runs/VIT2_npr_deb_full_s0 | 0.723573 | 0.7236 | có |
| hh  | ViT + HH  | runs/VIT2_hh_deb_full_s0  | 0.687395 | 0.6874 | có |
| wpt | ViT + WPT | runs/VIT2_wpt_deb_full_s0 | 0.541310 | 0.5413 | có |

## 3. Nhất quán với pipeline gốc

8 ảnh `demo_anh` × 4 mô hình = 32 cặp, `/api/predict` gọi qua HTTP thật, pipeline gốc chạy trong tiến trình riêng
(cwd = `AIGI_ROOT`), theo hai cách:

| Cách đối chiếu | Sai lệch tuyệt đối lớn nhất | Ghi chú |
|---|---|---|
| `python -m src.demo --runs <4 run> --images demo_anh --csv <file tạm>` | 4.70e-07 | CSV làm tròn 6 chữ số, nên đây chỉ là sai số làm tròn |
| Gọi thẳng `load_run` + `predict_detail`, điểm không làm tròn | **0 (0.00e+00)** | cả 32 cặp trùng từng bit |

Mục tiêu ≤ 1e-5: đạt ở cả hai. Nhãn, số cửa sổ, chiều rộng và chiều cao ảnh trùng pipeline gốc ở cả 32 cặp.

## 4. Ca lỗi (qua HTTP)

| Ca | Kết quả |
|---|---|
| WEBP hợp lệ (256×256, lossless) | 200 |
| PNG 16-bit (mode `I;16`) | 415 `unsupported_type` |
| File text đổi đuôi .jpg | 400 `invalid_image` |
| JPEG bị cắt cụt | 400 `invalid_image` |
| GIF đổi đuôi .jpg | 415 `unsupported_type` |
| BMP đổi đuôi .png | 415 `unsupported_type` |
| File rác đúng 10 MB (không vượt giới hạn) | 400 `invalid_image` |
| File 10 MB + 1 byte | 413 `file_too_large` |
| File 11 MB, có Content-Length | 413 `file_too_large` |
| File 11 MB gửi chunked, không Content-Length | 413 `file_too_large` |
| PNG 4097×8 px | 413 `file_too_large` |
| `model_id` lạ | 422 `unknown_model` |
| Thiếu `model_id` | 422 `unknown_model` |
| Thiếu `file` | 400 `invalid_image` |
| `GET /api/khongco` | 404 `http_error` |
| `GET /api/predict` (sai phương thức) | 405 `http_error` |

Sau các ca lỗi server vẫn trả 200 cho request hợp lệ, và thư mục tạm của server không còn file nào ở cuối lần chạy.
Thử tay (không nằm trong script): multipart thiếu boundary → 400 `http_error` "Yêu cầu không hợp lệ.";
`DELETE /api/models` → 405 `http_error` "Phương thức không được hỗ trợ.".

Chưa kiểm thử được: nhánh 500 `inference_failed` (không có cách kích hoạt tự nhiên; chỉ đọc code).

## 5. Quá tải và hàng chờ

Trong script (8 request đồng thời, ảnh 512 px, mô hình rgb):

- 3 request trả 200 (xong sau 1.2, 2.3, 3.5 giây), 5 request trả 503 `busy`; 503 chậm nhất sau 0.03 giây.
- `GET /` trả 200 sau 74 ms trong lúc quá tải; sau đợt đó server nhận request mới bình thường.

Thử tay lặp lại kịch bản của review (45 request đồng thời, ảnh 1024 px, mô hình rgb):

- 3 request trả 200 (sau 7.0, 13.6, 20.2 giây), 42 request trả 503; 503 chậm nhất sau 0.49 giây.
- `GET /` giữa đợt: 200 sau 83 ms. File tạm giữa đợt: 1. RSS giữa đợt 963 MB, đỉnh 1.14 GB (trước khi sửa: 1,7 GB,
  40 file tạm, `GET /` treo hơn 20 giây).

Thử tay client ngắt kết nối khi đang chờ: request A (1024 px) đang chạy, request B gửi rồi ngắt sau 1 giây, request C
(256 px) gửi sau. Log server ghi "Bỏ request: client đã ngắt kết nối trong lúc chờ lượt" cho B; C xong sau 6.0 giây,
tức ngay sau A, không phải chờ thêm một lượt suy luận của B.

## 6. Thời gian suy luận (CPU, 4 nhân)

`inference_ms` chỉ đo lời gọi `predict_detail` (không tính chờ lượt, kiểm tra upload, mạng). Mỗi ô: 1 lần làm nóng
rồi trung vị của 3 lần. Ảnh lấy nguyên từ `demo_anh`, không cắt/đổi kích thước.

| Mô hình | 256 px — AI_adm_584_adm_153.png (1 cửa sổ) | 512 px — AI_sd14_025_sdv4_00020.png (9 cửa sổ) | 1024 px — AI_mj_768_midjourney_197.png (49 cửa sổ) |
|---|---|---|---|
| rgb | 156 ms | 1256 ms | 6531 ms |
| npr | 143 ms | 1171 ms | 6413 ms |
| hh  | 153 ms | 1147 ms | 6574 ms |
| wpt | 155 ms | 1172 ms | 6679 ms |

Số dao động theo tải của máy (các lần chạy trước khi sửa cho 110–580 ms ở 256 px và 6,7–9,6 giây ở 1024 px với cùng
đoạn code suy luận); nên coi đây là bậc độ lớn, không phải benchmark.

- Khởi động + nạp 4 mô hình: 19.1 giây ở lần chạy này (các lần khác: 16.5 và 21.6 giây).
- RAM tiến trình server (RSS): 816 MB ngay sau khi nạp 4 mô hình; 896 MB sau khi phục vụ các request kiểm thử.
- `/api/health` trả lời sau 5 ms trong lúc đang suy luận ảnh 1024 px.

Đo tay ảnh lớn, mỗi phép 1 lần, mô hình rgb, ảnh là AI_mj_768 phóng to rồi lưu JPEG (chỉ để đo thời gian, điểm số
không có ý nghĩa):

| Ảnh | Số cửa sổ | inference_ms | Đỉnh RSS tiến trình |
|---|---|---|---|
| 2048×2048 | 225 | 28.8 giây | 1.23 GB |
| 4096×4096 | 961 | 135.9 giây | 1.84 GB |

(Trước khi sửa, cùng hai ảnh này đo được 35.1 và 152.2 giây — cùng đoạn code suy luận, khác tải máy.)
Thời gian tăng gần tuyến tính theo số cửa sổ, khoảng 130–150 ms mỗi cửa sổ.

## 7. Hạn chế

- Mô hình train trên ảnh sinh bởi SD 1.4 (GenImage); khả năng tổng quát sang bộ sinh khác không đồng đều
  (xem `KET_QUA_TOAN_ANH.md` trong `aigi_final`). Báo cáo này không tạo thêm số accuracy/benchmark nào.
- `score_ai` là sigmoid chưa hiệu chuẩn, không phải xác suất thật và không phải accuracy.
- Ngưỡng được chọn trên tập validation SD 1.4 (chỉ số Youden); có thể không tối ưu cho ảnh từ nguồn khác.
- Ảnh lớn chậm trên CPU (4096 px: hơn 2 phút). Request đã được nhận thì chờ tới lượt không giới hạn thời gian; xấu
  nhất phải chờ 2 ảnh phía trước, nên ba ảnh 4096 px liên tiếp có thể vượt timeout 5 phút của frontend.
- Suy luận đã bắt đầu thì không huỷ được, kể cả khi client ngắt kết nối giữa chừng (chỉ bỏ được request còn đang chờ).
- Hành vi ảnh đầu vào theo pipeline gốc: bỏ kênh alpha, không áp dụng EXIF orientation, ảnh động chỉ chấm khung đầu.
  Ảnh 16-bit/32-bit mỗi kênh bị từ chối (415) vì pipeline gốc bão hoà chúng thành ảnh gần trắng.
- Body vượt 10 MB được đọc bỏ tới tối đa 64 MB rồi mới trả 413 (để client nhận được phản hồi thay vì đứt kết nối).
- `final.pt` nạp bằng pickle (`weights_only=False`): chỉ dùng checkpoint tự tạo.
- Chỉ dành cho local, một người dùng: không xác thực, không rate limit ngoài hàng chờ 1 + 2.

## 8. Điểm lệch / bổ sung so với đặc tả

- Thiếu `file` → 400 `invalid_image`; thiếu `model_id` → 422 `unknown_model`; cạnh dài > `MAX_SIDE` (mặc định 4096 px, đổi bằng biến môi trường) → 413
  `file_too_large`; ảnh 16-bit → 415 `unsupported_type` (đặc tả không gán mã cho các ca này).
- Thêm hai mã ngoài đặc tả: 503 `busy` (quá tải hàng chờ, trả ngay) và `http_error` (404/405/400 của framework,
  thông báo tiếng Việt).
- "Timeout hợp lý cho CPU" của đặc tả: không còn timeout phía server. Thay bằng giới hạn hàng chờ (1 chạy + 2 chờ);
  timeout do client đặt.
- Chấp nhận thêm định dạng Pillow `MPO` (JPEG nhiều khung từ một số điện thoại), coi như JPEG.
- Ảnh được giải mã hai lần (một lần để xác thực, một lần trong `predict_detail`); không tính vào `inference_ms`.

## 9. Chưa làm / chưa kiểm được

- Dockerfile: **chưa build thử** (máy không có docker), kể cả `HEALTHCHECK` mới thêm. Giai đoạn build frontend giả
  định `frontend/package-lock.json` và `npm run build` tạo `frontend/dist`.
- Kiểm thử giao diện trên trình duyệt (mục 6.5): thuộc phần frontend. Ở đây chỉ xác minh `GET /` trả 200 với
  `frontend/dist` hiện có.
- Chưa thử nhánh CUDA (máy không có GPU); `load_run` tự chọn thiết bị qua `get_device()` của code gốc.
- `requirements.lock.txt` (tạo bằng `uv pip freeze` trên venv sạch) đã được cài lại bằng `uv` vào một venv sạch
  khác, ra đúng cùng bộ gói, và server chạy từ venv đó cho các phép đo tay ở mục 5–6. Chưa thử cài bằng `pip` thường.
  Lần chạy `check_all.py` dùng `~/aigi_web_venv`, có vài gói gián tiếp khác phiên bản với file khoá (idna, packaging,
  tqdm); torch, torchvision, timm, numpy, pillow, fastapi, starlette, uvicorn giống nhau.
- Hook aislop báo 2 lỗi `hallucinated-import` cho `from src.config ...` / `from src.evaluate ...` trong `app.py`:
  dương tính giả (gói `src` là pipeline gốc, nạp qua `sys.path` từ `AIGI_ROOT` theo yêu cầu đặc tả), để nguyên.

## 10. Lệnh chạy và URL đã xác minh

```bash
cd /mnt/d/xulyanh/web_demo/backend && ~/aigi_web_venv/bin/python app.py     # http://127.0.0.1:8000
~/aigi_web_venv/bin/python /mnt/d/xulyanh/web_demo/backend/tests/check_all.py   # kiểm thử (cổng rảnh tự chọn, tự tắt)
```

Đã xác minh: `GET /api/health` 200, `GET /` 200 (với `frontend/dist` hiện có), `POST /api/predict` 200.
Mọi server thử đã được tắt sau khi kiểm thử.
