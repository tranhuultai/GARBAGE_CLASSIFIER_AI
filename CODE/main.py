"""main.py - mo thang giao dien demo, khong menu. Chay tu thu muc CODE/."""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # phai dat truoc khi import keras lan dau

# pylint: disable=wrong-import-position
import importlib.util
import sys
import subprocess

MODEL_PATH = "models/best_model.keras"


def main():
    """Mo giao dien demo - kiem tra nhanh model/Streamlit roi chay thang, khong hoi gi them."""
    if not os.path.exists(MODEL_PATH):
        print(f"[X] Khong tim thay {MODEL_PATH} - chay 'python src/train.py' truoc.")
        sys.exit(1)

    if importlib.util.find_spec("streamlit") is None:
        print("[X] Chua cai Streamlit. Chay 'pip install -r requirements.txt' truoc.")
        sys.exit(1)

    print("Dang mo giao dien demo...")
    # python -m streamlit de khong phu thuoc PATH
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/app.py"], check=False)


if __name__ == "__main__":
    main()
