"""test_train.py - kiem tra train.py, dung du lieu gia thay vi dataset that de test nhanh.
Neu data_processing.py chua co tren nhanh nay thi bo qua ca file (xem importorskip ben duoi)."""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

# pylint: disable=wrong-import-position
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("data_processing", reason="data_processing.py chua co tren nhanh nay")

from data_processing import ImageFolderDataset, CLASS_NAMES  # pylint: disable=import-error
import train  # pylint: disable=import-error


def _tao_du_lieu_gia(thu_muc_goc, so_anh_moi_lop=3):
    """Tao processed/{train,val,test}/<lop> voi vai anh mau ngau nhien, du de train 1 epoch."""
    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            class_dir = Path(thu_muc_goc, split, class_name)
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(so_anh_moi_lop):
                pixels = np.random.randint(0, 255, (224, 224, 3), dtype="uint8")
                Image.fromarray(pixels).save(class_dir / f"anh_{i}.jpg")


@pytest.fixture(name="du_lieu_gia")
def fixture_du_lieu_gia(tmp_path, monkeypatch):
    """Tao du lieu gia, tro build_datasets cua train.py toi thu muc tam nay."""
    _tao_du_lieu_gia(tmp_path)

    def build_datasets_gia(batch_size=32):
        return (
            ImageFolderDataset(str(tmp_path / "train"), batch_size=batch_size),
            ImageFolderDataset(str(tmp_path / "val"), batch_size=batch_size, shuffle=False),
            ImageFolderDataset(str(tmp_path / "test"), batch_size=batch_size, shuffle=False),
        )

    monkeypatch.setattr(train, "build_datasets", build_datasets_gia)
    return tmp_path


def test_main_khong_co_finetune_goi_train_baseline(monkeypatch):
    """main() khong truyen '--finetune' phai goi train_baseline(), khong goi finetune()."""
    da_goi = {"train_baseline": False, "finetune": False}
    monkeypatch.setattr(train, "train_baseline", lambda: da_goi.update(train_baseline=True))
    monkeypatch.setattr(train, "finetune", lambda: da_goi.update(finetune=True))

    train.main([])

    assert da_goi["train_baseline"] is True
    assert da_goi["finetune"] is False


def test_main_co_finetune_goi_finetune(monkeypatch):
    """main() co truyen '--finetune' phai goi finetune(), khong goi train_baseline()."""
    da_goi = {"train_baseline": False, "finetune": False}
    monkeypatch.setattr(train, "train_baseline", lambda: da_goi.update(train_baseline=True))
    monkeypatch.setattr(train, "finetune", lambda: da_goi.update(finetune=True))

    train.main(["--finetune"])

    assert da_goi["finetune"] is True
    assert da_goi["train_baseline"] is False


def test_train_baseline_chay_va_luu_model(du_lieu_gia, tmp_path, monkeypatch):
    # pylint: disable=unused-argument
    """train_baseline() phai chay het 1 epoch va luu duoc file model that.
    du_lieu_gia khong dung truc tiep - fixture nay chi de tao du lieu gia + monkeypatch san."""
    model_path = tmp_path / "best_model.keras"
    monkeypatch.setattr(train, "BEST_MODEL_PATH", str(model_path))

    train.train_baseline(epochs=1)

    assert model_path.exists()


def test_finetune_chay_duoc_tren_model_da_co(du_lieu_gia, tmp_path, monkeypatch):
    # pylint: disable=unused-argument
    """finetune() phai nap duoc model co san, chay 1 epoch, khong crash.
    du_lieu_gia khong dung truc tiep - fixture nay chi de tao du lieu gia + monkeypatch san."""
    from model import build_model  # pylint: disable=import-error,import-outside-toplevel

    model_path = tmp_path / "best_model.keras"
    build_model(num_classes=len(CLASS_NAMES)).save(model_path)
    monkeypatch.setattr(train, "BEST_MODEL_PATH", str(model_path))

    train.finetune(epochs=1)

    assert model_path.exists()
