"""app.py - giao dien Streamlit: chup/tai anh, model du doan, goi y xu ly."""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # phai dat truoc khi import keras lan dau

# pylint: disable=wrong-import-position
import numpy as np
import streamlit as st
from keras.models import load_model
from PIL import Image

MODEL_PATH = "models/best_model.keras"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 60  # duoi nguong nay thi canh bao khong chac chan

# Thu tu nay phai khop dung voi CLASS_NAMES trong src/data_processing.py (thu tu luc train)
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

CLASS_LABELS_VI = {
    "cardboard": "Bìa carton",
    "glass": "Thủy tinh",
    "metal": "Kim loại",
    "paper": "Giấy",
    "plastic": "Nhựa",
    "trash": "Rác thải chung",
}

CLASS_ICONS = {
    "cardboard": "📦",
    "glass": "🍾",
    "metal": "🥫",
    "paper": "📄",
    "plastic": "🧴",
    "trash": "🗑️",
}

DISPOSAL_GUIDE = {
    "cardboard": [
        "Gấp phẳng thùng carton để tiết kiệm diện tích.",
        "Giữ khô ráo, không dính dầu mỡ hay thức ăn thừa.",
        "Bỏ vào thùng rác tái chế.",
    ],
    "paper": [
        "Không tái chế giấy dính dầu mỡ hoặc giấy ăn đã dùng.",
        "Gỡ ghim bấm, băng keo trước khi bỏ nếu có.",
        "Bỏ vào thùng rác tái chế cùng với bìa carton.",
    ],
    "glass": [
        "Rửa sạch chai lọ trước khi bỏ đi.",
        "Nếu bị vỡ, bọc lại bằng giấy báo trước khi bỏ để tránh gây thương tích.",
        "Không bỏ chung với gương hay bóng đèn - đây là loại kính khác.",
    ],
    "metal": [
        "Rửa sạch lon nước ngọt, hộp thực phẩm trước khi bỏ.",
        "Có thể ép dẹp lon để tiết kiệm diện tích thùng rác.",
        "Kim loại tái chế được gần như vô hạn lần, rất đáng để phân loại riêng.",
    ],
    "plastic": [
        "Kiểm tra ký hiệu tái chế (số 1-7) in dưới đáy sản phẩm.",
        "Rửa sạch, để ráo nước trước khi bỏ vào thùng tái chế.",
        "Tháo nắp chai ra riêng nếu nắp khác loại nhựa với thân chai.",
    ],
    "trash": [
        "Không tái chế được - bỏ vào thùng rác thông thường.",
        "Cân nhắc chọn sản phẩm ít bao bì hơn cho lần mua sau.",
    ],
}

st.set_page_config(page_title="Phân Loại Rác Thải AI", page_icon="♻️", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Nen toan trang: mot lop gradient mint rat nhat o tren cung, con lai trang */
[data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(180deg, rgba(16, 185, 129, 0.05) 0%, rgba(255,255,255,0) 320px);
}

.main-title {
    background: -webkit-linear-gradient(45deg, #0ba360, #3cba92);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.6rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0;
}
.subtitle { text-align: center; color: #64748b; margin-bottom: 2rem; }

/* Sidebar: nen mint nhat, tach biet ro voi vung noi dung chinh */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f0fdf6 0%, #ffffff 100%);
    border-right: 1px solid rgba(16, 185, 129, 0.15);
}
.class-chip {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0.8rem;
    margin: 0.3rem 0;
    border-radius: 10px;
    background: #ffffff;
    border: 1px solid rgba(16, 185, 129, 0.18);
    font-weight: 600;
    font-size: 0.95rem;
}

/* Khung upload/camera: bo goc + vien mint de dong bo voi theme */
div[data-testid="stTabs"] {
    background: #ffffff;
    border: 1px solid rgba(16, 185, 129, 0.15);
    border-radius: 15px;
    padding: 1rem 1.2rem 1.4rem;
}

.result-box {
    padding: 1.8rem;
    border-radius: 15px;
    background: rgba(16, 185, 129, 0.06);
    border: 1px solid rgba(16, 185, 129, 0.25);
    text-align: center;
}
.result-label { font-size: 1.8rem; font-weight: 700; margin: 0.4rem 0; color: #0ba360; }

.guide-box {
    padding: 1.2rem 1.4rem;
    margin-top: 0.8rem;
    border-radius: 15px;
    background: #fffbeb;
    border: 1px solid rgba(245, 158, 11, 0.3);
}
.guide-box p { margin: 0 0 0.5rem; font-weight: 700; color: #92400e; }
.guide-box ul { margin: 0; padding-left: 1.2rem; color: #78350f; }
.guide-box li { margin-bottom: 0.3rem; }

/* Trang thai rong: chua co anh nao duoc chon */
.empty-state {
    padding: 3rem 1.5rem;
    border-radius: 15px;
    border: 2px dashed rgba(16, 185, 129, 0.3);
    text-align: center;
    color: #64748b;
}
.empty-state .empty-icon { font-size: 2.6rem; margin-bottom: 0.6rem; }

/* Doi mau thanh do tin cay tu xanh duong mac dinh sang xanh la dong bo theme */
[data-testid="stProgressBarTrack"] { background-color: rgba(16, 185, 129, 0.15) !important; }
[data-testid="stProgressBarTrack"] > div { background-color: #0ba360 !important; }

/* Bo goc anh xem truoc cho dong bo voi cac card khac */
[data-testid="stImage"] img { border-radius: 12px; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    """Nap model, cache lai giua cac lan chay (Streamlit tu goi lai script moi thao tac)."""
    return load_model(MODEL_PATH)


def predict(image: Image.Image):
    """Tra ve (nhan du doan, do tin cay %, xac suat ca 6 lop) cho 1 anh PIL."""
    image = image.convert("RGB").resize(IMG_SIZE)
    # Khong tu chuan hoa o day - model da co san lop preprocess_input ben trong (xem model.py)
    img_array = np.expand_dims(np.array(image, dtype="float32"), axis=0)

    model = get_model()
    # type: ignore - type stub cua keras.models.load_model chua day du, da kiem chung luc runtime
    probabilities = model.predict(img_array, verbose=0)[0]  # type: ignore
    predicted_index = int(np.argmax(probabilities))
    return CLASS_NAMES[predicted_index], float(probabilities[predicted_index]) * 100, probabilities


def render_sidebar():
    """Sidebar: gioi thieu ngan gon + danh muc 6 lop model nhan dien duoc."""
    with st.sidebar:
        st.markdown("## ♻️ Garbage Classifier AI")
        st.caption("Đồ án môn AI - phân loại rác thải bằng transfer learning MobileNetV2.")
        st.markdown("### Model nhận diện được 6 loại:")
        for class_name in CLASS_NAMES:
            st.markdown(
                f"<div class='class-chip'>{CLASS_ICONS[class_name]} "
                f"{CLASS_LABELS_VI[class_name]}</div>",
                unsafe_allow_html=True,
            )


def render_result(predicted_class, confidence, probabilities):
    """Hien thi ket qua du doan: nhan, do tin cay, goi y xu ly, chi tiet xac suat."""
    icon = CLASS_ICONS[predicted_class]
    label_vi = CLASS_LABELS_VI[predicted_class]
    st.markdown(
        f"""
        <div class='result-box'>
            <div style='font-size: 3rem;'>{icon}</div>
            <div style='color: #64748b;'>Được phân loại là</div>
            <div class='result-label'>{label_vi}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(int(confidence), 100))
    st.caption(f"Độ tin cậy: {confidence:.1f}%")

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(
            "⚠️ Độ tin cậy thấp - model không chắc chắn. Vật thể có thể không thuộc rõ "
            "1 trong 6 loại đã học, hoặc ảnh chưa đủ rõ. Thử chụp lại ảnh rõ hơn, đủ "
            "sáng, chỉ chụp 1 món rác duy nhất trong khung hình."
        )
    else:
        st.success("✅ Model khá chắc chắn với kết quả này.")

    tips_html = "".join(f"<li>{tip}</li>" for tip in DISPOSAL_GUIDE[predicted_class])
    st.markdown(
        f"<div class='guide-box'><p>💡 Cách xử lý</p><ul>{tips_html}</ul></div>",
        unsafe_allow_html=True,
    )

    with st.expander("Xem chi tiết xác suất từng lớp"):
        st.bar_chart(
            {CLASS_LABELS_VI[name]: float(p) for name, p in zip(CLASS_NAMES, probabilities)}
        )


def main():
    """Man hinh chinh Streamlit: chup/tai anh -> du doan -> hien thi ket qua."""
    render_sidebar()
    st.markdown(
        "<div class='main-title'>Phân Loại Rác Thải Bằng AI ♻️</div>", unsafe_allow_html=True
    )
    st.markdown(
        "<div class='subtitle'>Chụp hoặc tải lên 1 ảnh rác thải để AI nhận diện loại rác</div>",
        unsafe_allow_html=True,
    )

    try:
        get_model()
    # pylint: disable-next=broad-exception-caught
    except Exception:  # model thieu/hong khong duoc lam sap app luc demo
        st.error(f"Không nạp được model ở `{MODEL_PATH}`. Chạy `python src/train.py` trước.")
        return

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("### 📸 Ảnh rác thải")
        tab_camera, tab_upload = st.tabs(["Chụp ảnh", "Tải ảnh lên"])
        with tab_camera:
            camera_file = st.camera_input("Chụp 1 món rác")
        with tab_upload:
            uploaded_file = st.file_uploader(
                "Chọn ảnh (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"]
            )

        image_file = camera_file or uploaded_file
        image = Image.open(image_file) if image_file is not None else None
        if image is not None:
            st.image(image, caption="Ảnh đã chọn", use_container_width=True)

    with col_result:
        st.markdown("### 📊 Kết quả")
        if image is None:
            st.markdown(
                "<div class='empty-state'><div class='empty-icon'>♻️</div>"
                "Chụp hoặc tải 1 ảnh rác thải lên để xem kết quả phân loại.</div>",
                unsafe_allow_html=True,
            )
        else:
            predicted_class, confidence, probabilities = predict(image)
            render_result(predicted_class, confidence, probabilities)


if __name__ == "__main__":
    main()
