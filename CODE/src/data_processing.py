"""data_processing.py - tai dataset Kaggle, chia train/val/test, tao dataset cho Keras."""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # phai dat truoc khi import keras lan dau

# pylint: disable=wrong-import-position
import math
import random
import shutil
from pathlib import Path

import numpy as np
import keras
from PIL import Image

DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
KAGGLE_DATASET = "asdasdasasdas/garbage-classification"

# Thu tu danh sach nay quyet dinh chi so nhan (0..5) luc train - app.py va evaluate.py
# phai dung dung thu tu nay thi nhan du doan moi khop.
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

IMG_SIZE = (224, 224)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15  # phan con lai (0.15) danh cho test
SEED = 42


def _raw_data_ready():
    """Kiem tra data/raw/ da co du 6 lop, moi lop it nhat 1 anh, chua."""
    for class_name in CLASS_NAMES:
        class_dir = Path(DATA_RAW_DIR, class_name)
        if not class_dir.is_dir() or not any(class_dir.glob("*")):
            return False
    return True


def download_dataset():
    """Tai dataset tu Kaggle ve data/raw/<lop>/. Bo qua neu da co san du lieu."""
    if _raw_data_ready():
        print(f"Da co anh trong {DATA_RAW_DIR}/, bo qua buoc tai.")
        return

    print(f"Dang tai dataset Kaggle '{KAGGLE_DATASET}' ...")
    try:
        import kagglehub  # pylint: disable=import-outside-toplevel
        downloaded_path = kagglehub.dataset_download(KAGGLE_DATASET)
    except Exception as error:
        raise RuntimeError(
            "Khong tai duoc dataset tu dong qua kagglehub.\n"
            "Cach khac phuc:\n"
            "  1) Tao API token tai kaggle.com/settings -> Create New Token -> tai kaggle.json.\n"
            "  2) Dat file kaggle.json vao thu muc ~/.kaggle/ "
            "(hoac %USERPROFILE%/.kaggle/ tren Windows).\n"
            "  3) Chay lai 'python src/data_processing.py'.\n"
            "  Hoac tai thu cong tai kaggle.com/datasets/asdasdasasdas/garbage-classification, "
            f"giai nen roi tu copy anh tung lop vao {DATA_RAW_DIR}/<ten lop>/."
        ) from error

    print(f"Dang sap xep anh vao {DATA_RAW_DIR}/<lop>/ ...")
    count_by_class = {name: 0 for name in CLASS_NAMES}
    for file_path in Path(downloaded_path).rglob("*"):
        if not file_path.is_file():
            continue
        class_name = file_path.parent.name.lower()
        if class_name in CLASS_NAMES:
            dest_dir = Path(DATA_RAW_DIR, class_name)
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_dir / file_path.name)
            count_by_class[class_name] += 1

    missing = [name for name, count in count_by_class.items() if count == 0]
    if missing:
        raise RuntimeError(f"Khong tim thay anh nao cho cac lop: {missing} trong dataset da tai.")
    print("Tai va sap xep du lieu xong.")


def _count_processed():
    counts = {}
    for class_name in CLASS_NAMES:
        counts[class_name] = {
            split_name: len(list(Path(DATA_PROCESSED_DIR, split_name, class_name).glob("*")))
            for split_name in ("train", "val", "test")
        }
    return counts


def _already_split():
    """True neu ca 3 tap train/val/test da co anh (khong chi rieng train/)."""
    return all(
        Path(DATA_PROCESSED_DIR, split_name).is_dir()
        and any(Path(DATA_PROCESSED_DIR, split_name).rglob("*.*"))
        for split_name in ("train", "val", "test")
    )


def split_dataset():
    """Chia anh raw/ thanh processed/{train,val,test}/<lop> theo ty le 70/15/15."""
    if _already_split():
        print(
            f"Da chia du lieu roi ({DATA_PROCESSED_DIR}/ da co du train/val/test) - bo qua. "
            f"Xoa thu muc {DATA_PROCESSED_DIR}/ neu muon chia lai tu dau."
        )
        return _count_processed()

    random.seed(SEED)
    for class_name in CLASS_NAMES:
        images = sorted(Path(DATA_RAW_DIR, class_name).glob("*"))
        if not images:
            raise RuntimeError(
                f"Khong co anh trong {DATA_RAW_DIR}/{class_name}/ - chay download_dataset() truoc."
            )
        random.shuffle(images)

        n_train = int(len(images) * TRAIN_RATIO)
        n_val = int(len(images) * VAL_RATIO)
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }
        for split_name, image_paths in splits.items():
            dest_dir = Path(DATA_PROCESSED_DIR, split_name, class_name)
            dest_dir.mkdir(parents=True, exist_ok=True)
            for source_path in image_paths:
                shutil.copy2(source_path, dest_dir / source_path.name)

    return _count_processed()


class ImageFolderDataset(keras.utils.PyDataset):
    """Doc anh tu <thu_muc>/<lop>/*.jpg theo tung batch - khong dung TensorFlow.
    Augmentation KHONG nam o day - no nam trong model.py (cac lop Random*)."""

    def __init__(self, directory, batch_size=32, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Moi phan tu la 1 cap (duong dan anh, chi so lop) - duyet qua tung lop,
        # trong moi lop duyet qua tung anh, roi gop lai thanh 1 danh sach.
        self.samples = []
        for class_index, class_name in enumerate(CLASS_NAMES):
            image_paths = sorted(Path(directory, class_name).glob("*"))
            for image_path in image_paths:
                self.samples.append((image_path, class_index))

        self.indices = np.arange(len(self.samples))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.samples) / self.batch_size)

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        images = np.zeros((len(batch_indices), *IMG_SIZE, 3), dtype="float32")
        labels = np.zeros((len(batch_indices), len(CLASS_NAMES)), dtype="float32")
        for i, sample_index in enumerate(batch_indices):
            path, class_index = self.samples[sample_index]
            image = Image.open(path).convert("RGB").resize(IMG_SIZE)
            images[i] = np.array(image, dtype="float32")  # [0,255] - model tu preprocess
            labels[i, class_index] = 1.0
        return images, labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def build_datasets(batch_size=32):
    """Tra ve (train_dataset, val_dataset, test_dataset) tu data/processed/."""
    train_dataset = ImageFolderDataset(f"{DATA_PROCESSED_DIR}/train", batch_size=batch_size)
    val_dataset = ImageFolderDataset(
        f"{DATA_PROCESSED_DIR}/val", batch_size=batch_size, shuffle=False
    )
    test_dataset = ImageFolderDataset(
        f"{DATA_PROCESSED_DIR}/test", batch_size=batch_size, shuffle=False
    )
    return train_dataset, val_dataset, test_dataset


def main():
    """Chay 'python src/data_processing.py': tai dataset roi chia train/val/test."""
    print("=== Buoc 1: Tai dataset ===")
    download_dataset()

    print("=== Buoc 2: Chia train/val/test (70/15/15) ===")
    counts = split_dataset()

    print(f"{'Lop':<12}{'train':>8}{'val':>8}{'test':>8}")
    total = {"train": 0, "val": 0, "test": 0}
    for class_name, class_counts in counts.items():
        n_train = class_counts["train"]
        n_val = class_counts["val"]
        n_test = class_counts["test"]
        print(f"{class_name:<12}{n_train:>8}{n_val:>8}{n_test:>8}")
        total["train"] += n_train
        total["val"] += n_val
        total["test"] += n_test
    print(f"{'TONG':<12}{total['train']:>8}{total['val']:>8}{total['test']:>8}")
    print(f"Xong. Du lieu san sang o {DATA_PROCESSED_DIR}/.")
    return counts


if __name__ == "__main__":
    main()
