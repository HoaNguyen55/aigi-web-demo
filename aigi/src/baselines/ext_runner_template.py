"""MẪU: chạy một baseline bằng code/checkpoint gốc của tác giả trên danh sách ảnh test, ghi CSV path,score.

Chép file này vào thư mục repo của tác giả (chạy trong môi trường Python của repo đó), sửa 2 hàm load_model / predict
theo README của tác giả, rồi:
  python ext_runner_template.py --list ../aigi-detect/ext_lists/genimage_all_val.txt \
      --out ../aigi-detect/ext_scores/NPR/genimage_all_val.csv --batch 64
GIỮ ĐÚNG tiền xử lý của tác giả (crop/resize/chuẩn hoá) — không dùng tiền xử lý của bài để công bằng với họ.
"""
import argparse
import csv
from pathlib import Path

import torch
from PIL import Image


def load_model(device):
    """TODO: dựng mô hình + nạp checkpoint theo README của tác giả. Trả về (model, transform)."""
    raise NotImplementedError("Sửa hàm này theo repo của tác giả")


@torch.no_grad()
def predict(model, batch):
    """TODO: trả về tensor [B] = xác suất ẢNH GIẢ (hoặc logit). Nhiều repo ra logit -> torch.sigmoid."""
    return torch.sigmoid(model(batch).flatten())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tf = load_model(dev)
    model.eval()
    paths = Path(a.list).read_text(encoding="utf-8").splitlines()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if Path(a.out).exists():  # chạy lại thì tiếp tục
        with open(a.out, encoding="utf-8") as f:
            done = {r["path"] for r in csv.DictReader(f)}
    todo = [p for p in paths if p not in done]
    with open(a.out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["path", "score"])
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in chunk]).to(dev)
            for p, s in zip(chunk, predict(model, x).float().cpu().tolist()):
                w.writerow([p, s])
            print(f"{i + len(chunk)}/{len(todo)}", end="\r")
    print(f"\nĐã ghi {a.out}")


if __name__ == "__main__":
    main()
