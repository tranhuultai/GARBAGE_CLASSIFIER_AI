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


def finetune(epochs=FINETUNE_EPOCHS):
    """Mo dong bang vai lop cuoi MobileNetV2, train tiep voi learning rate rat nho."""
    train_ds, val_ds, _ = build_datasets()

    print(f"Dang nap model da train o {BEST_MODEL_PATH} de fine-tune...")
    model = load_model(BEST_MODEL_PATH)
    # type: ignore - type stub keras.models.load_model chua day du (xem app.py)
    _, baseline_val_accuracy = model.evaluate(val_ds, verbose=0)  # type: ignore
    print(f"Val_accuracy cua model baseline (truoc fine-tune): {baseline_val_accuracy:.4f}")
    unfreeze_for_finetune(model, num_layers=30, learning_rate=1e-5)

    callbacks = [
        # initial_value_threshold = accuracy baseline - chi ghi de best_model.keras neu
        # fine-tune that su tot hon, khong de fine-tune te hon ghi de mat ban baseline tot.
        ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            initial_value_threshold=baseline_val_accuracy,
            verbose=1,
        ),
        EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1),
    ]

    print(f"Bat dau fine-tune: {epochs} epoch, learning rate rat nho (1e-5).")
    model.fit(  # type: ignore
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )
    print(
        f"Fine-tune xong. {BEST_MODEL_PATH} chi bi ghi de neu fine-tune dat val_accuracy cao "
        f"hon {baseline_val_accuracy:.4f} (baseline) - neu khong, file van la ban baseline cu.\n"
        "Kiem tra lai bang 'python src/evaluate.py'."
    )


def main(argv=None):
    """Chay 'python src/train.py' (baseline) hoac them '--finetune' de fine-tune."""
    argv = sys.argv[1:] if argv is None else argv
    if "--finetune" in argv:
        finetune()
    else:
        train_baseline()


if __name__ == "__main__":
    main()
