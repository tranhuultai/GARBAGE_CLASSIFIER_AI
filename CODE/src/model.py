"""model.py - xay kien truc model transfer learning MobileNetV2 (xem PROJECT_SPEC.md muc 5.4)."""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # phai dat truoc khi import keras (xem CLAUDE.md)

# pylint: disable=wrong-import-position
import numpy as np
import keras
from keras import layers, Model
from keras.applications import MobileNetV2
