"""evaluate.py - danh gia model: accuracy, classification report, confusion matrix."""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

from data_processing import build_datasets, CLASS_NAMES

BEST_MODEL_PATH = "models/best_model.keras"
CONFUSION_MATRIX_PATH = "docs/confusion_matrix.png"

def _save_confusion_matrix(y_true, y_pred):
    """Ve va luu confusion matrix vao CONFUSION_MATRIX_PATH."""
    os.makedirs(os.path.dirname(CONFUSION_MATRIX_PATH), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=CLASS_NAMES, ax=ax, cmap="Blues", xticks_rotation=45
    )
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH)
    plt.close(fig)
    print(f"Da luu confusion matrix vao {CONFUSION_MATRIX_PATH}")
    print("Doc: hang la nhan that, cot la nhan du doan, duong cheo la du doan dung.")

def main(argv=None):
    """Chay 'python src/evaluate.py [duong_dan_model]' (mac dinh models/best_model.keras)."""
    argv = sys.argv[1:] if argv is None else argv
    model_path = argv[0] if argv else BEST_MODEL_PATH

    print(f"Dang nap model tu {model_path} ...")
    model = load_model(model_path)

    _, _, test_ds = build_datasets()
    print(f"Dang du doan tren {len(test_ds.samples)} anh test ...")
    predictions = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.array([class_index for _, class_index in test_ds.samples])

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy tong the tren tap test: {acc * 100:.2f}%\n")

    print("Precision / Recall / F1 theo tung lop:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    _save_confusion_matrix(y_true, y_pred)

if __name__ == "__main__":
    main()