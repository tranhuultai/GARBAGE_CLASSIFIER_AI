import cv2
import numpy as np
import tensorflow as tf  # Hoặc dùng `import torch` tùy thuộc vào khung làm việc của nhóm

class GarbageClassifier:
    def __init__(self, model_path="model_garbage.h5"):
        # 1. Tải Model AI đã huấn luyện
        self.model = tf.keras.models.load_model(model_path)
        # Định nghĩa kích thước ảnh đầu vào mà Model yêu cầu (ví dụ: 224x224)
        self.img_size = (224, 224)
        # Danh sách các nhãn phân loại (điều chỉnh theo danh sách của nhóm)
        self.class_names = ['Carton', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

    def preprocess_image(self, image_input):
        """
        Tiền xử lý hình ảnh đầu vào
        """
        # Nếu truyền vào đường dẫn file ảnh (str) -> Đọc ảnh bằng OpenCV
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_input

        # Resize ảnh về chuẩn đầu vào của Model
        img_resized = cv2.resize(img, self.img_size)
        
        # Chuẩn hóa giá trị điểm ảnh (Pixel) về khoảng [0, 1]
        img_normalized = img_resized / 255.0
        
        # Thêm chiều Batch (Batch size = 1) -> (1, 224, 224, 3)
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch

    def predict(self, image_input):
        """
        Dự đoán nhãn rác thải từ hình ảnh
        """
        # Tiền xử lý ảnh
        processed_img = self.preprocess_image(image_input)
        
        # Đưa ảnh qua Model để dự đoán
        predictions = self.model.predict(processed_img)
        
        # Lấy chỉ số có độ tin cậy cao nhất
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        label = self.class_names[class_idx]

        return {
            "label": label,
            "confidence": round(confidence * 100, 2)
        }

# --- TEST CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # Khởi tạo pipeline
    classifier = GarbageClassifier(model_path="CODE/model.h5")
    
    # Dự đoán một ảnh thử nghiệm
    result = classifier.predict("test_image.jpg")
    print(f"Loại rác: {result['label']} | Độ chính xác: {result['confidence']}%")
