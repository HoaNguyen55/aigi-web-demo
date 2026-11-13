# TASK: Web demo local – phát hiện ảnh AI với 4 mô hình ViT đơn đặc trưng

## 1. Mục tiêu
Xây dựng web demo chạy local: tải ảnh → chọn 1 trong 4 mô hình → nhận nhãn AI-Generated / Real Image.
Phạm vi: 4 mô hình đơn (RGB, NPR, HH, WPT). KHÔNG có mô hình kết hợp. KHÔNG triển khai công khai ở bước này.

## 2. Đầu vào có sẵn
- Source + checkpoint gốc (chỉ đọc, không sửa): /mnt/d/xulyanh/extracted/aigi_final
- Giao diện đã thiết kế bằng Open Design: <ĐƯỜNG DẪN THƯ MỤC XUẤT TỪ OPEN DESIGN>
- Thư mục làm việc mới: /mnt/d/xulyanh/web_demo  (backend/, frontend/, README.md, Dockerfile)

## 3. Mô hình và pipeline (đã kiểm chứng từ source – dùng đúng, không suy đoán lại)
| model_id | Run | Ngưỡng τ |
|---|---|---|
| rgb | runs/VIT1_rgb_deb_full_s0 | 0,8870 |
| npr | runs/VIT2_npr_deb_full_s0 | 0,7236 |
| hh  | runs/VIT2_hh_deb_full_s0  | 0,6874 |
| wpt | runs/VIT2_wpt_deb_full_s0 | 0,5413 |
- Nạp: src.evaluate.load_run(cfg, run). Suy luận: src.evaluate.predict_detail(model, cfg, [path]).
- Ngưỡng: đọc từ checkpoint (rcfg["_val_threshold"]), không hardcode; bảng trên chỉ để đối chiếu.
- Đầu vào: ảnh RGB kích thước bất kỳ, [0,1], KHÔNG resize, KHÔNG chuẩn hoá mean/std; đệm phản chiếu (pad_full);
  cửa sổ 256×256 bước 128 px, dùng mọi cửa sổ; attention pooling → 1 logit → sigmoid = P(AI).
- Nhãn: "ai" nếu P(AI) ≥ τ, ngược lại "real". score_real = 1 − score_ai.
- Bắt buộc gọi lại code gốc (import từ aigi_final/src); không viết lại trích đặc trưng, kiến trúc hay tiền xử lý.
- Nếu checkpoint không nạp được hoặc thiếu file: dừng, báo cáo nguyên nhân, không tạo kết quả giả.

## 4. Backend (FastAPI + PyTorch CPU)
- Môi trường: venv riêng; chỉ cài phụ thuộc cần cho suy luận (torch CPU, torchvision, timm, numpy, pillow, pyyaml,
  fastapi, uvicorn, python-multipart + những gì import của src/evaluate.py thực sự cần). Ghi requirements có phiên bản.
- Nạp 4 mô hình một lần khi khởi động, tái sử dụng giữa các request; torch.inference_mode(); tự chọn CUDA nếu có.
- Endpoint:
  - GET /api/health → { "ok": true, "models_loaded": [...] }
  - GET /api/models → [{ id, name, description, threshold }]
  - POST /api/predict (multipart: file, model_id) →
    { model_id, model_name, label, score_ai, score_real, threshold, inference_ms, image: {width, height}, n_windows }
  - Lỗi: { "error": { "code", "message" } } với mã unsupported_type (415), file_too_large (413),
    invalid_image (400), unknown_model (422), inference_failed (500).
- Giới hạn: JPG/JPEG/PNG/WEBP, tối đa 10 MB, cạnh dài tối đa 4096 px (từ chối nếu vượt; không tự thu nhỏ ảnh vì
  sẽ làm đổi kết quả). Kiểm tra bằng nội dung file (Pillow), không tin đuôi file.
- Ảnh tải lên: ghi file tạm, xoá ngay sau khi suy luận; không ghi log nội dung ảnh hay tên file.
- Mỗi lần chỉ chạy một suy luận (khoá), timeout hợp lý cho CPU. Không secrets, không API key.

## 5. Frontend
- Dùng giao diện xuất từ Open Design; chỉ thay dữ liệu mẫu bằng lời gọi API thật và nối các trạng thái
  (trống, có ảnh, đang phân tích, kết quả, lỗi). Không thiết kế lại.
- Backend phục vụ luôn bản build tĩnh của frontend (một cổng, không cần CORS).
- Không trình bày score như accuracy.

## 6. Kiểm thử (bắt buộc, báo cáo số thật)
1. Nạp được cả 4 checkpoint; ngưỡng đọc ra khớp bảng mục 3.
2. Nhất quán với pipeline gốc: chạy `python -m src.demo --runs <run> --images demo_anh/ --csv ...` và so P(AI)
   với /api/predict trên cùng 8 ảnh demo_anh × 4 mô hình; sai lệch tuyệt đối ≤ 1e-5.
3. Ca lỗi: file không phải ảnh, sai định dạng, > 10 MB, model_id sai → đúng mã lỗi.
4. Đo thời gian suy luận từng mô hình trên ảnh 256, 512, 1024 px (CPU).
5. Giao diện: thử trên trình duyệt ở desktop và mobile, đủ các trạng thái.
Không tự tạo accuracy/benchmark; số liệu đánh giá mô hình chỉ được trích từ KET_QUA_TOAN_ANH.md nếu cần.

## 7. Bàn giao
- Source backend + frontend, requirements, Dockerfile (CPU), README: cài đặt, chạy local, cập nhật checkpoint.
- Báo cáo ngắn: kiến trúc, checkpoint dùng, kết quả kiểm thử nhất quán, thời gian suy luận, hạn chế
  (ví dụ: mô hình train trên SD1.4; điểm chưa hiệu chuẩn; ngưỡng chọn trên validation SD1.4).
- Lệnh chạy và URL local đã xác minh.

## 8. Nguyên tắc
Tuần tự: xác minh nạp checkpoint → backend → đối chiếu với pipeline gốc → nối frontend → kiểm thử.
Ưu tiên đúng và tái lập được hơn hiệu ứng giao diện. Thiếu thông tin thì báo cáo, không đoán.
