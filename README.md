# GARBAGE_CLASSIFIER_AI

Ứng dụng phân loại rác thải bằng hình ảnh (transfer learning MobileNetV2 + Streamlit).

## Cấu trúc repo

- `CODE/` — toàn bộ mã nguồn (chỉ chứa code, không chứa tài liệu).
- `DOC/` — báo cáo đồ án (`DOC/report.md`).
- `SLIDES/` — slide thuyết trình.
- `EXTRA/` — tài liệu/ảnh phụ, gồm `EXTRA/demo_images/` (ảnh dự phòng cho lúc bảo vệ).
- `requirements.txt` — thư viện cần cài, dùng chung cho toàn bộ `CODE/`.

## Bắt đầu nhanh

```bash
pip install -r requirements.txt
cd CODE
python src/data_processing.py    # tải dataset Kaggle, chia train/val/test
python src/train.py              # huấn luyện model, lưu models/best_model.keras
python src/evaluate.py           # accuracy, F1 theo lớp, confusion matrix
python main.py                   # DÙNG LÚC DEMO THẬT — mở thẳng giao diện Streamlit
```

Fine-tune (tùy chọn, sau khi bản baseline đã ổn): `python src/train.py --finetune`.

Mọi lệnh chạy từ trong `CODE/`.

## Yêu cầu môi trường

- Dùng Keras 3 chạy trên PyTorch (không dùng TensorFlow) — cài được trên mọi bản Python
  hiện đại, kể cả bản rất mới mà TensorFlow chưa hỗ trợ (TensorFlow thường chỉ hỗ trợ tới
  các bản Python cũ hơn).
- Sau khi train xong, **commit `CODE/models/best_model.keras`** — giảng viên chạy trực tiếp
  `python main.py`, không train lại từ đầu.
