# Mô hình so sánh — cách đưa vào bài cho công bằng

Giao thức của bài: **chỉ train trên GenImage SD1.4**, test trên các bộ sinh khác. Mỗi baseline thuộc một trong 4 cách xử lý:

| Ký hiệu | Cách xử lý | Ghi trong bài |
|---|---|---|
| **A** | Dùng checkpoint chính thức đã train trên GenImage SD1.4 | so sánh trực tiếp |
| **B** | Train lại trên SD1.4 (trong code này, hoặc bằng code gốc) | so sánh trực tiếp |
| **C** | Dùng checkpoint chính thức train trên dữ liệu riêng của tác giả | đánh dấu "†: dữ liệu train gốc của tác giả" |
| **D** | Chỉ chép số trong bài báo | đánh dấu "‡: số do tác giả báo cáo", không đưa vào DeLong |

Thông tin dưới đây được kiểm tra từ README/code chính thức (10/2026). Mục ghi *chưa xác minh* cần kiểm tra lại khi tải.

| # | Mô hình | Repo | Checkpoint chính thức train trên | Cách xử lý | Trong code này |
|---|---|---|---|---|---|
| 1 | CNNSpot (CVPR'20) | PeterWang512/CNNDetection (CC BY-NC-SA) | ProGAN 20 lớp | **B** (rẻ) | `B_cnnspot`: ResNet-50 + blur/JPEG, train lại SD1.4 |
| 2 | UnivFD (CVPR'23) | WisconsinAIVision/UniversalFakeDetect (MIT) | ProGAN | **B** | `B_univfd`: CLIP ViT-L/14 đóng băng + lớp tuyến tính |
| 3 | NPR (CVPR'24) | chuangchuangtan/NPR-DeepfakeDetection | ProGAN 4 lớp **và GenImage SD1.4** (Google Drive) | **A** | `EXT_NPR` (+ `F_npr_*` là đặc trưng NPR trên thân của bài) |
| 4 | FreqNet (AAAI'24) | chuangchuangtan/FreqNet-DeepfakeDetection | ProGAN 4 lớp | **B** bằng code gốc, hoặc **D** | `EXT_FreqNet` |
| 5 | SAFE (KDD'25) | Ouxiang-Li/SAFE (Apache-2.0) | ProGAN 4 lớp | **B** bằng code gốc (+C) | `EXT_SAFE` |
| 6 | C2P-CLIP (AAAI'25) | chuangchuangtan/C2P-CLIP-DeepfakeDetection | GenImage SD1.4 (*chưa xác minh nội dung thư mục*) | **A** (dự phòng B: `scripts/train_genimage.sh`) | `EXT_C2P-CLIP` |
| 7 | AIDE (ICLR'25) | shilinyan99/AIDE (MIT) | thư mục Drive "AIDE_checkpoints" (*chưa xác minh có SD1.4*) | **A** nếu có SD1.4, không thì **B** | `EXT_AIDE` |
| 8 | DRCT (ICML'24) | beibuwandeluori/DRCT | **GenImage SD1.4** (ConvNeXt-B, CLIP) trên ModelScope | **A** | `EXT_DRCT` |
| 9 | Aligned (ICLR'25) | anisundar18/AlignedForensics | COCO/LSUN + tái tạo VAE | **C** + bản train lại | `B_aligned` (B) + `EXT_Aligned` (C) |
| 10 | B-Free (CVPR'25) | grip-unina/B-Free (chỉ phi lợi nhuận) | COCO + SD2.1 inpainting | **C** | `EXT_BFree` |
| 11 | DDA (NeurIPS'25) | roy-ch/Dual-Data-Alignment (Apache-2.0) | COCO + tái tạo VAE (HF `Junwei-Xi/Dual-Data-Alignment`) | **C** | `EXT_DDA` |

## Phương pháp mới 2024–2026 (bổ sung để reviewer không hỏi "sao không so với X")

| # | Mô hình | Repo / giấy phép | Checkpoint train trên | Cách xử lý | Trong code này |
|---|---|---|---|---|---|
| 12 | **Effort** (ICML'25) | YZY-stack/Effort-AIGI-Detection (CC BY-NC 4.0) | **GenImage SD1.4** (+ ProGAN) | **A** | `EXT_Effort` + số đã công bố (Table 9: mAcc in 91.1, 8 ô trung bình 90.09) |
| 13 | **CO-SPY** (CVPR'25) | Megum1/Co-Spy (MIT) | DRCT-2M SD1.4 (không phải GenImage) ; ProGAN | **B** (train lại SD1.4) hoặc A kèm chú thích | `EXT_CO-SPY` |
| 14 | **OmniAID** (ICML'26) | yunncheng/OmniAID (MIT) | **GenImage SD1.4** (CLIP-L/14@336) | **A** | `EXT_OmniAID` |
| 15 | **VIB-Net** (CVPR'25) | oceanzhf/VIBAIGCDetect (chưa ghi giấy phép) | SD1.4 và ProGAN (*chi tiết chưa xác minh*) | **A** | `EXT_VIB-Net` |
| 16 | **SPAI** (CVPR'25) | mever-team/spai (Apache-2.0) | ảnh LDM của Corvi et al. | **C** (mạnh trên Synthbuster, ảnh mọi độ phân giải) | `EXT_SPAI` |
| 17 | **RINE** (ECCV'24) | mever-team/rine (Apache-2.0) | ProGAN 1/2/4 lớp; LDM | **C** cho giao thức 1, **A** cho giao thức 2 (ProGAN 4 lớp) | `EXT_RINE` |
| 18 | **FatFormer** (CVPR'24) | (chưa xác minh) | ProGAN 4 lớp | **D** ở giao thức 1 (C2P-CLIP chạy lại), A ở giao thức 2 | số đã công bố |
| 19 | **PPL** (ICLR'26) | chưa thấy code | GenImage SD1.4 | **D** (Table 1: 97.2 CLIP / 95.9 DINOv2) | số đã công bố (chỉ trung bình) |
| 20 | **IAPL** (CVPR'26) | chưa thấy code | GenImage SD1.4 ; ProGAN | **D** (Table 3: 96.7) | số đã công bố (chỉ trung bình) |

Số liệu đã công bố của ~40 phương pháp nằm trong `reported/` (xem `reported/README.md`) và tự gộp vào bảng
so sánh bằng `python -m src.compare`.

## Quy trình cho baseline chạy bằng code gốc (EXT_*)

1. Chạy `python run_all.py --stages 5 --only EXPORT` → danh sách ảnh test ở `ext_lists/<danh sách>.txt`
   (`genimage_all_val`, `genimage_hard4_val`, `test_synthbuster`, `test_openfake`, ...).
2. Clone repo của tác giả, tải checkpoint, tạo môi trường riêng cho repo đó (thư viện có thể khác phiên bản).
3. Chạy mô hình trên **đúng** các ảnh trong file `.txt`, giữ **đúng tiền xử lý của tác giả** (kích thước crop,
   chuẩn hoá). Ghi CSV hai cột `path,score` — `score` là xác suất ảnh giả (logit cũng được).
   Mẫu vòng lặp: `src/baselines/ext_runner_template.py`.
4. Đặt CSV vào `ext_scores/<TÊN>/<danh sách>.csv`, ví dụ `ext_scores/NPR/genimage_all_val.csv`.
5. Chạy `python run_all.py --stages 5` → baseline xuất hiện như run `EXT_<TÊN>` trong mọi bảng (cùng độ đo,
   cùng tách seen/unseen) và được kiểm định DeLong với phương pháp chính.

Lưu ý công bằng:
- Không chỉnh tham số của baseline dùng checkpoint (A, C): giữ nguyên như tác giả công bố.
- Baseline train lại (B) dùng **cùng** dữ liệu, cùng tập VAL và cùng lưới tinh chỉnh lr như phương pháp chính
  (`src/tune.py`, mục `baselines`).
- DDA ghi chú kết quả phụ thuộc phiên bản libjpeg khi đọc ảnh: ghi phiên bản Pillow trong bài.
