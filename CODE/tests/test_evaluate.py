"""test_evaluate.py - kiem tra evaluate.py: ve confusion matrix, chay main() tren model gia.
Neu data_processing.py chua co tren nhanh nay thi bo qua ca file (xem importorskip ben duoi)."""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

# pylint: disable=wrong-import-position
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("data_processing", reason="data_processing.py chua co tren nhanh nay")

from data_processing import CLASS_NAMES  # pylint: disable=import-error
import evaluate  # pylint: disable=import-error


def test_save_confusion_matrix_tao_file(tmp_path, monkeypatch):
    """_save_confusion_matrix() phai ve va luu duoc file anh."""
    matrix_path = tmp_path / "confusion_matrix.png"
    monkeypatch.setattr(evaluate, "CONFUSION_MATRIX_PATH", str(matrix_path))

    # phai co du ca 6 nhan (0..5) thi display_labels=CLASS_NAMES (6 ten) moi khop so luong
    y_true = np.array([0, 1, 2, 3, 4, 5, 0, 1])
    y_pred = np.array([0, 1, 1, 3, 4, 5, 0, 2])
    # pylint: disable=protected-access
    evaluate._save_confusion_matrix(y_true, y_pred)

    assert matrix_path.exists()


def test_main_chay_duoc_va_ve_duoc_confusion_matrix(tmp_path, monkeypatch):
    """main() phai nap model, du doan tren tap test gia, va ve confusion matrix - khong crash."""
    # pylint: disable=import-error,import-outside-toplevel
    from model import build_model
    from data_processing import ImageFolderDataset

    for class_name in CLASS_NAMES:
        class_dir = Path(tmp_path, "test", class_name)
        class_dir.mkdir(parents=True, exist_ok=True)
        pixels = np.random.randint(0, 255, (224, 224, 3), dtype="uint8")
        Image.fromarray(pixels).save(class_dir / "anh_0.jpg")

    def build_datasets_gia(batch_size=32):
        test_ds = ImageFolderDataset(str(tmp_path / "test"), batch_size=batch_size, shuffle=False)
        return None, None, test_ds

    monkeypatch.setattr(evaluate, "build_datasets", build_datasets_gia)
    monkeypatch.setattr(evaluate, "CONFUSION_MATRIX_PATH", str(tmp_path / "confusion_matrix.png"))

    model_path = tmp_path / "test_model.keras"
    build_model(num_classes=len(CLASS_NAMES)).save(model_path)

    evaluate.main([str(model_path)])

    assert (tmp_path / "confusion_matrix.png").exists()
