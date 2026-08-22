import streamlit as st
from PIL import Image
import time
import random

# Cấu hình trang cơ bản
st.set_page_config(
    page_title="Hệ Thống Phân Loại Rác AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thêm CSS tuỳ chỉnh để làm giao diện "Premium"
st.markdown("""
<style>
    /* Tổng quan Font chữ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Hiệu ứng gradient cho tiêu đề chính */
    .main-title {
        background: -webkit-linear-gradient(45deg, #0ba360, #3cba92);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0px;
        padding-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.2rem;
        margin-bottom: 40px;
    }

    /* Box kết quả AI */
    .result-box {
        padding: 25px;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
        text-align: center;
        animation: fadeIn 0.8s ease-in-out;
    }
    
    .result-label {
        font-size: 2rem;
        font-weight: 700;
        margin: 10px 0;
    }
    
    /* Animation mượt mà */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    /* Tuỳ chỉnh Button tải lên */
    .stFileUploader > div > div > div > button {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stFileUploader > div > div > div > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    
    /* Tuỳ chỉnh Button phân tích */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 2rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Giao diện Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3299/3299935.png", width=100)
    st.markdown("## 🌿 AI Eco-Sorter")
    st.markdown("Hệ thống sử dụng Trí tuệ nhân tạo (Deep Learning) để tự động nhận dạng và phân loại rác thải vào các nhóm chính xác.")
    
    st.markdown("### Danh mục nhận diện:")
    st.markdown("""
    - 🍎 **Rác hữu cơ** (Thức ăn thừa, lá cây...)
    - ♻️ **Rác tái chế** (Nhựa, giấy, kim loại...)
    - ⚠️ **Rác độc hại** (Pin, linh kiện điện tử...)
    - 🗑️ **Rác vô cơ khác**
    """)
    
    st.markdown("---")
    st.caption("Phiên bản 1.0.0 | 2026")

# Giao diện chính
st.markdown("<div class='main-title'>Nhận Diện Phân Loại Rác AI ♻️</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Cùng chung tay bảo vệ môi trường bằng cách phân loại rác đúng cách</div>", unsafe_allow_html=True)

# Layout 2 cột
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📸 Tải ảnh rác thải lên")
    uploaded_file = st.file_uploader("Kéo thả hoặc chọn file ảnh (JPG, PNG, JPEG)", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Hiển thị ảnh
        image = Image.open(uploaded_file)
        st.image(image, caption='Ảnh đã tải lên', use_container_width=True)
        
        # Nút phân tích
        analyze_button = st.button("🔍 Phân tích ngay", use_container_width=True)

with col2:
    st.markdown("### 📊 Kết quả phân tích")
    
    # Placeholder chờ kết quả
    if uploaded_file is None:
        st.info("👈 Vui lòng tải một hình ảnh rác thải lên ở cột bên trái để AI tiến hành phân loại.")
        st.image("https://illustrations.popsy.co/amber/environment.svg", width=300)
    else:
        if 'analyze_button' in locals() and analyze_button:
            # Mô phỏng quá trình AI đang chạy
            with st.spinner("AI đang quét và phân tích hình ảnh..."):
                time.sleep(2) # Giả lập thời gian load
                
            # Random kết quả để Demo (Trong thực tế sẽ gọi API Model AI ở đây)
            categories = {
                "Tái chế (Recyclable)": {"color": "#10b981", "icon": "♻️", "desc": "Chai nhựa, giấy báo, vỏ lon. Vui lòng làm sạch trước khi bỏ vào thùng tái chế!"},
                "Hữu cơ (Organic)": {"color": "#84cc16", "icon": "🍎", "desc": "Thức ăn thừa, vỏ trái cây. Có thể dùng làm phân xanh cho cây trồng."},
                "Độc hại (Hazardous)": {"color": "#ef4444", "icon": "⚠️", "desc": "Pin, bóng đèn, hóa chất. CẦN vứt ở khu vực thu gom rác thải nguy hại đặc biệt!"},
                "Khác (Residual)": {"color": "#64748b", "icon": "🗑️", "desc": "Tã, giấy ăn đã qua sử dụng, gốm sứ vỡ. Vứt vào thùng rác sinh hoạt thông thường."}
            }
            
            result_key = random.choice(list(categories.keys()))
            result_data = categories[result_key]
            confidence = random.uniform(85.5, 99.9)
            
            # Hiển thị kết quả đẹp mắt
            st.markdown(f"""
            <div class='result-box'>
                <div style='font-size: 4rem; margin-bottom: 10px;'>{result_data["icon"]}</div>
                <div style='font-size: 1.2rem; color: #94a3b8;'>Được phân loại là:</div>
                <div class='result-label' style='color: {result_data["color"]}'>{result_key}</div>
                <p style='color: #cbd5e1; margin-top: 15px;'>{result_data["desc"]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Hiển thị thanh Confidence Score
            st.markdown("#### Độ tin cậy của AI (Confidence Score)")
            st.progress(int(confidence))
            st.markdown(f"<div style='text-align: right; color: #10b981; font-weight: bold;'>{confidence:.2f}%</div>", unsafe_allow_html=True)
            
            st.success("✅ Phân tích thành công!")
            
            # Thông tin thêm (Metrics)
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("Thời gian xử lý", f"{random.uniform(0.1, 0.8):.2f}s")
            with col2_2:
                st.metric("Model version", "v1.0.4-beta")
        else:
            st.warning("Nhấn nút **'🔍 Phân tích ngay'** bên trái để xem kết quả.")
