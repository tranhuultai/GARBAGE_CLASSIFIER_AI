"""test_model.py - kiem tra kien truc model.py (build_model, unfreeze_for_finetune).
Khong can du lieu that hay model da train - chi kiem tra kien truc, shape, hanh vi dong bang."""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

# pylint: disable=wrong-import-position
import numpy as np
import pytest

# pylint: disable=import-error
# import-error: pylint khong biet conftest.py da them src/ vao sys.path luc chay pytest
from model import build_model, unfreeze_for_finetune, _find_base_model

NUM_CLASSES = 6


@pytest.fixture(scope="module", name="model")
def fixture_model():
    """Build 1 lan dung chung cho ca file test - build_model ton thoi gian (tai MobileNetV2)."""
    return build_model(num_classes=NUM_CLASSES)


def test_output_shape(model):
    """Model phai tra ve dung (so_anh, so_lop) cho moi batch dau vao."""
    dummy_images = np.zeros((2, 224, 224, 3), dtype="float32")
    output = model.predict(dummy_images, verbose=0)
    assert output.shape == (2, NUM_CLASSES)


def test_output_la_phan_phoi_xac_suat():
    """Dau ra softmax phai la xac suat: >= 0 va tong moi anh xap xi 1."""
    fresh_model = build_model(num_classes=NUM_CLASSES)
    dummy_image = np.random.rand(1, 224, 224, 3).astype("float32") * 255
    output = fresh_model.predict(dummy_image, verbose=0)
    assert (output >= 0).all()
    assert np.isclose(output.sum(), 1.0, atol=1e-4)


def test_base_model_dong_bang_mac_dinh(model):
    """build_model() phai dong bang MobileNetV2 - chi train dau phan loai luc dau."""
    base_model = _find_base_model(model)
    assert base_model.trainable is False


def test_unfreeze_for_finetune_mo_dung_so_lop():
    """unfreeze_for_finetune(num_layers=N) chi duoc mo dung N lop cuoi cua base model."""
    fresh_model = build_model(num_classes=NUM_CLASSES)
    unfreeze_for_finetune(fresh_model, num_layers=10, learning_rate=1e-5)

    base_model = _find_base_model(fresh_model)
    assert base_model.trainable is True
    so_lop_dang_train = sum(1 for layer in base_model.layers if layer.trainable)
    assert so_lop_dang_train == 10


def test_find_base_model_bao_loi_khi_khong_co():
    """_find_base_model() phai raise ValueError neu model khong co layer MobileNetV2 nao."""
    import keras  # pylint: disable=import-outside-toplevel

    model_khong_co_base = keras.Sequential([keras.layers.Dense(6, input_shape=(10,))])
    with pytest.raises(ValueError):
        _find_base_model(model_khong_co_base)
