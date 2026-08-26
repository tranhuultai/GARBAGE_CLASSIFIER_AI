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
