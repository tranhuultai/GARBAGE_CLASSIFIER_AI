# GARBAGE_CLASSIFIER_AI

Đồ án phân loại rác thải bằng ảnh, dùng transfer learning MobileNetV2 kết hợp giao diện
Streamlit. Chụp hoặc tải lên 1 ảnh rác, app trả về 1 trong 6 loại (bìa carton, thủy tinh,
kim loại, giấy, nhựa, rác thải chung) kèm độ tin cậy và gợi ý xử lý.

Model hiện tại: ~85% val accuracy, ~81% test accuracy (xem `CODE/docs/confusion_matrix.png`
sau khi chạy `evaluate.py`).

## Cấu trúc repo

- `CODE/` - code (`src/` xử lý dữ liệu + model + train + đánh giá, `app/` giao diện,
  `tests/` test tự động, `models/` model đã train, `docs/` biểu đồ/confusion matrix sinh ra)
- `DOC/` - báo cáo đồ án
- `SLIDES/` - slide thuyết trình
- `EXTRA/` - ảnh demo dự phòng và tài liệu phụ
- `requirements.txt` - danh sách thư viện cần cài

## Cài đặt & chạy thử

```bash
pip install -r requirements.txt
cd CODE
python src/data_processing.py    # tải dataset, chia train/val/test
python src/train.py              # train model, lưu vào models/best_model.keras
python src/evaluate.py           # xem accuracy và confusion matrix
python main.py                   # mở app demo
```

Muốn fine-tune thì chạy thêm `python src/train.py --finetune`.

Nhớ chạy mọi lệnh ở trên trong thư mục `CODE/`, không thì import sẽ lỗi.

`models/best_model.keras` đã có sẵn trong repo, nên có thể bỏ qua bước tải dữ liệu/train và
chạy thẳng `python main.py` để xem demo.

## Chạy test

```bash
cd CODE
pytest tests/
```

Test dùng ảnh giả (random) để chạy nhanh, không cần dataset thật. Cùng bộ lệnh này chạy tự
động trên GitHub Actions mỗi lần push (xem `.github/workflows/ci.yml`).

## Lưu ý môi trường

Dùng Keras 3 chạy trên PyTorch, không dùng TensorFlow.

Sau khi train xong phải commit `CODE/models/best_model.keras` lên repo - máy khác pull về
chạy thẳng `python main.py` sẽ báo lỗi thiếu model nếu file này chưa có.
