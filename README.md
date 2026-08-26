# GARBAGE_CLASSIFIER_AI

Đồ án phân loại rác thải bằng ảnh, dùng transfer learning MobileNetV2 kết hợp giao diện Streamlit.

## Cấu trúc repo

- `CODE/` - code
- `DOC/` - báo cáo (`DOC/report.md`)
- `SLIDES/` - slide thuyết trình
- `EXTRA/` - ảnh demo dự phòng và tài liệu phụ
- `requirements.txt` - danh sách thư viện cần cài

## Chạy thử

```bash
pip install -r requirements.txt
cd CODE
python src/data_processing.py    # tải dataset, chia train/val/test
python src/train.py              # train model, luu vao models/best_model.keras
python src/evaluate.py           # xem accuracy va confusion matrix
python main.py                   # mo app demo
```

Muốn fine-tune thì chạy thêm `python src/train.py --finetune`.

Nhớ chạy mọi lệnh ở trên trong thư mục `CODE/`, không thì import sẽ lỗi.

## Lưu ý môi trường

Dùng Keras 3 chạy trên PyTorch chứ không dùng TensorFlow, vì lúc setup máy Python bản mới
cài TensorFlow không được (TensorFlow thường chỉ hỗ trợ vài bản Python cũ hơn).

Sau khi train xong phải commit `CODE/models/best_model.keras` lên repo - máy khác pull về
chạy thẳng `python main.py` sẽ báo lỗi thiếu model nếu file này chưa có.
