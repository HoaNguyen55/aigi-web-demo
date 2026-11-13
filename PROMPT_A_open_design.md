# Thiết kế giao diện: Web demo phát hiện ảnh do AI tạo

## Mục tiêu
Thiết kế MỘT trang web (single page) cho demo học thuật: người dùng tải một ảnh, chọn một trong bốn mô hình
và xem kết quả "AI-Generated" hay "Real Image". Chỉ thiết kế frontend; backend đã có API riêng (mục Hợp đồng API).

## Phong cách
- Modern minimalist dashboard, nền trắng, màu nhấn xanh dương, chữ sans-serif, nhiều khoảng trắng.
- Chuyên nghiệp, không hiệu ứng thừa. Ngôn ngữ giao diện: tiếng Việt, giữ nguyên nhãn "AI-Generated" / "Real Image".

## Bố cục
- Desktop (≥1280×720): toàn bộ chức năng nằm trong một màn hình, không cuộn. Hai cột:
  - Trái: vùng tải ảnh (kéo thả hoặc bấm chọn), danh sách 4 mô hình (chọn một), nút "Analyze Image".
  - Phải: ảnh xem trước và thẻ kết quả.
- Tablet và mobile: hai cột xếp dọc, cho phép cuộn.

## Thành phần
1. Vùng tải ảnh: nhận JPG, JPEG, PNG, WEBP, tối đa 10 MB. Sau khi có ảnh, có nút "Đổi ảnh".
2. Chọn mô hình (4 lựa chọn, mỗi lựa chọn có tên và một dòng mô tả):
   - ViT + RGB — giá trị màu gốc (đường cơ sở)
   - ViT + NPR — chênh lệch giữa các điểm ảnh lân cận
   - ViT + HH — băng chéo wavelet Haar (nhiễu tần số cao)
   - ViT + WPT — 16 dải tần wavelet packet
3. Nút "Analyze Image": vô hiệu khi chưa có ảnh; khi đang xử lý hiện trạng thái loading.
4. Thẻ kết quả:
   - Nhãn lớn: "AI-Generated" hoặc "Real Image".
   - "Điểm AI" (0–1) kèm thanh ngang có vạch đánh dấu ngưỡng của mô hình.
   - Tỷ lệ hai lớp: P(AI) và P(Real) = 1 − P(AI).
   - Tên mô hình, ngưỡng quyết định, thời gian suy luận (ms), kích thước ảnh, số cửa sổ đã chấm.
   - Chú thích nhỏ: "Điểm AI là đầu ra của mô hình, không phải độ chính xác."

## Trạng thái cần thiết kế
Trống (chưa có ảnh) · đã có ảnh, chưa phân tích · đang phân tích · có kết quả AI · có kết quả Real ·
lỗi (sai định dạng, quá dung lượng, mất kết nối, lỗi suy luận) với thông báo rõ ràng và cách thử lại.

## Hợp đồng API (dùng đúng tên trường; trong bản thiết kế dùng dữ liệu mẫu theo đúng cấu trúc này)
- GET /api/models → [{ "id": "rgb", "name": "ViT + RGB", "description": "...", "threshold": 0.887 }, ...]
- POST /api/predict (multipart: file, model_id) →
  { "model_id": "npr", "model_name": "ViT + NPR", "label": "ai" | "real", "score_ai": 0.93, "score_real": 0.07,
    "threshold": 0.7236, "inference_ms": 1840, "image": { "width": 512, "height": 512 }, "n_windows": 9 }
- Lỗi → { "error": { "code": "unsupported_type" | "file_too_large" | "invalid_image" | "inference_failed",
                     "message": "..." } }

## Đầu ra mong muốn
React + Tailwind CSS, một trang, mọi lời gọi API gom vào một module để sau này thay dữ liệu mẫu bằng API thật.
Không thêm đăng nhập, không thêm trang phụ, không thêm thư viện biểu đồ.

## Không làm
Không hiển thị mô hình kết hợp (Fusion). Không bịa số liệu accuracy/benchmark. Không gọi "điểm AI" là độ chính xác.
