"""train.py - huan luyen model baseline, va (tuy chon, dung --finetune) fine-tune."""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # phai dat truoc khi import keras lan dau

# pylint: disable=wrong-import-position
import sys

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight

from data_processing import build_datasets, CLASS_NAMES
from model import build_model, unfreeze_for_finetune

BEST_MODEL_PATH = "models/best_model.keras"
BASELINE_EPOCHS = 15
FINETUNE_EPOCHS = 5


def train_baseline(epochs=BASELINE_EPOCHS):
    """Train model tu dau (base MobileNetV2 dong bang), luu model tot nhat vao best_model.keras."""
    train_ds, val_ds, _ = build_datasets()
    model = build_model(num_classes=len(CLASS_NAMES))

    # Neu 1 lop co it anh hon han cac lop khac, class_weight se phat nang hon loi o lop do
    # de model khong "bo quen" lop thieu so.
    sample_classes = np.array([class_index for _, class_index in train_ds.samples])
    unique_classes = np.unique(sample_classes)
    weights = compute_class_weight(
        class_weight="balanced", classes=unique_classes, y=sample_classes
    )
    # zip (khong phai enumerate) - de dung ngay ca khi 1 lop nao do vang mat trong tap train
    class_weight = dict(zip(unique_classes, weights))

    callbacks = [
        ModelCheckpoint(BEST_MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    ]

    print(
        f"Bat dau train baseline: {epochs} epoch, "
        f"{len(train_ds.samples)} anh train, {len(val_ds.samples)} anh validation."
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )
    print(f"Da luu model tot nhat (theo val_accuracy) vao {BEST_MODEL_PATH}")
