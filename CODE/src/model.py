"""model.py - xay kien truc model transfer learning MobileNetV2 (frozen base + GAP + Dense)."""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # phai dat truoc khi import keras lan dau

# pylint: disable=wrong-import-position
import numpy as np
import keras
from keras import layers, Model
from keras.applications import MobileNetV2


def build_model(num_classes, input_shape=(224, 224, 3)):
    """Xay model: augmentation + MobileNetV2 dong bang + GAP + Dropout + Dense softmax."""
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False  # dong bang pretrained weights, chi train dau phan loai

    inputs = keras.Input(shape=input_shape)
    # Augmentation - la layer trong model nen Keras tu bat luc fit(), tat luc predict()
    x = layers.RandomRotation(0.06)(inputs)
    x = layers.RandomTranslation(0.1, 0.1)(x)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomZoom(0.1)(x)
    x = keras.applications.mobilenet_v2.preprocess_input(x)  # scale [0,255] -> [-1,1]
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",  # nhan one-hot, khop voi labels trong data_processing.py
        metrics=["accuracy"],
    )
    return model


def _find_base_model(model):
    """Tim layer MobileNetV2 nam ben trong model (no duoc gan vao nhu 1 layer con)."""
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    raise ValueError("Khong tim thay base model (MobileNetV2) ben trong model.")


def unfreeze_for_finetune(model, num_layers=30, learning_rate=1e-5):
    """Mo dong bang num_layers cuoi cua MobileNetV2, recompile voi learning rate rat nho."""
    base_model = _find_base_model(model)
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    """Kiem tra nhanh: build model voi 6 lop, in summary, thu predict voi anh gia."""
    model = build_model(num_classes=6)
    model.summary()

    dummy_image = np.zeros((1, 224, 224, 3), dtype="float32")
    output = model.predict(dummy_image, verbose=0)  # type: ignore
    print(f"Output shape: {output.shape} (ky vong (1, 6)), tong xac suat: {output.sum():.4f}")


if __name__ == "__main__":
    main()
