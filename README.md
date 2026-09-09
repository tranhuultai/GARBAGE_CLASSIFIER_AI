# GARBAGE_CLASSIFIER_AI

Phân loại rác thải qua ảnh bằng transfer learning MobileNetV2, giao diện Streamlit. Chụp
hoặc tải lên 1 ảnh, app trả về 1 trong 6 loại rác kèm độ tin cậy và gợi ý xử lý.

## Yêu cầu

- Python 3.10+
- Keras 3 chạy trên backend PyTorch (không dùng TensorFlow)

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

Mọi lệnh chạy trong thư mục `CODE/`.

```bash
cd CODE
python main.py                   # mở demo - model đã có sẵn, không cần chạy các bước dưới
```

Muốn tự tải dữ liệu và train lại từ đầu:

```bash
python src/data_processing.py    # tải dataset, chia train/val/test
python src/train.py              # train model, lưu vào models/best_model.keras
python src/train.py --finetune   # fine-tune model đã train (tùy chọn)
python src/evaluate.py           # accuracy, F1, confusion matrix
```

## Kiểm thử

```bash
cd CODE
pytest tests/
```

Dùng ảnh giả để chạy nhanh, không cần dataset thật. Cùng bộ lệnh này chạy tự động trên
GitHub Actions ở mỗi lần push (`.github/workflows/ci.yml`).

## Cấu trúc dự án

| Thư mục | Nội dung |
|---|---|
| `CODE/src/` | Xử lý dữ liệu, kiến trúc model, huấn luyện, đánh giá |
| `CODE/app/` | Giao diện Streamlit |
| `CODE/tests/` | Test tự động |
| `CODE/models/` | Model đã huấn luyện |
| `CODE/docs/` | Confusion matrix, biểu đồ huấn luyện (sinh ra khi chạy) |
| `DOC/` | Báo cáo đồ án |
| `SLIDES/` | Slide thuyết trình |
| `EXTRA/` | Ảnh demo dự phòng, tài liệu phụ |

## Kết quả

Model hiện tại đạt khoảng 85% validation accuracy, 81% test accuracy trên 6 lớp (bìa
carton, thủy tinh, kim loại, giấy, nhựa, rác thải chung). Chi tiết ở
`CODE/docs/confusion_matrix.png`.
