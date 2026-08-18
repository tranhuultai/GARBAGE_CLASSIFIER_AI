import tkinter as tk
from tkinter import filedialog, ttk
try:
    from PIL import Image, ImageTk
except ImportError:
    print("Vui lòng cài đặt thư viện Pillow để hiển thị ảnh: pip install Pillow")
    import sys
    sys.exit(1)

class GarbageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Phân Loại Rác Thải")
        self.root.geometry("600x700")
        self.root.configure(bg="#0f172a") # Giao diện Dark mode
        
        # Font chữ
        title_font = ("Helvetica", 24, "bold")
        normal_font = ("Helvetica", 12)
        btn_font = ("Helvetica", 12, "bold")
        
        # Header
        self.header = tk.Label(root, text="EcoVision AI", font=title_font, bg="#0f172a", fg="#10b981")
        self.header.pack(pady=(30, 5))
        
        self.subtitle = tk.Label(root, text="Hệ thống phân loại rác thải bằng Python", font=normal_font, bg="#0f172a", fg="#94a3b8")
        self.subtitle.pack(pady=(0, 20))

        # Khung chứa ảnh
        self.preview_frame = tk.Frame(root, bg="#1e293b", bd=0, width=400, height=300)
        self.preview_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        self.preview_frame.pack_propagate(False) # Giữ cố định kích thước khung
        
        self.image_label = tk.Label(self.preview_frame, text="Chưa có ảnh nào được chọn\n\nNhấn 'Chọn Ảnh' bên dưới", bg="#1e293b", fg="#94a3b8", font=normal_font)
        self.image_label.pack(expand=True)
        
        # Khung chứa nút bấm
        self.btn_frame = tk.Frame(root, bg="#0f172a")
        self.btn_frame.pack(pady=20)

        # Nút chọn ảnh
        self.browse_btn = tk.Button(self.btn_frame, text="Chọn Ảnh", font=btn_font, bg="#3b82f6", fg="white", activebackground="#2563eb", activeforeground="white", command=self.browse_image, relief=tk.FLAT, cursor="hand2")
        self.browse_btn.grid(row=0, column=0, padx=10, ipadx=20, ipady=8)

        # Nút phân loại (ban đầu bị vô hiệu hóa)
        self.classify_btn = tk.Button(self.btn_frame, text="Phân Loại Ngay", font=btn_font, bg="#10b981", fg="white", activebackground="#059669", activeforeground="white", state=tk.DISABLED, command=self.classify, relief=tk.FLAT, cursor="hand2")
        self.classify_btn.grid(row=0, column=1, padx=10, ipadx=20, ipady=8)

        # Khung hiển thị kết quả
        self.result_frame = tk.Frame(root, bg="#0f172a")
        self.result_frame.pack(pady=10, fill=tk.X)
        
        self.result_title = tk.Label(self.result_frame, text="", font=("Helvetica", 18, "bold"), bg="#0f172a")
        self.result_title.pack()
        
        self.result_desc = tk.Label(self.result_frame, text="", font=normal_font, bg="#0f172a", fg="#f8fafc", wraplength=500)
        self.result_desc.pack(pady=10)

        self.current_image_path = None

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp")])
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            # Kích hoạt nút phân loại
            self.classify_btn.config(state=tk.NORMAL)
            # Xóa kết quả cũ
            self.result_title.config(text="")
            self.result_desc.config(text="")

    def display_image(self, path):
        try:
            img = Image.open(path)
            # Thay đổi kích thước ảnh để vừa với khung (giữ nguyên tỷ lệ)
            img.thumbnail((450, 350))
            self.tk_image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.tk_image, text="")
        except Exception as e:
            self.image_label.config(image='', text=f"Lỗi khi tải ảnh:\n{e}")

    def classify(self):
        if not self.current_image_path:
            return
            
        # Trạng thái UI đang xử lý
        self.classify_btn.config(state=tk.DISABLED, text="Đang phân tích...")
        self.root.update() # Cập nhật giao diện ngay lập tức
        
        # ---------------------------------------------------------
        # TODO: THÊM LOGIC MODEL AI CỦA BẠN VÀO ĐÂY
        # Ví dụ:
        # import torch
        # prediction = model.predict(self.current_image_path)
        # ---------------------------------------------------------
        
        # Thiết lập kết quả giao diện (Hiện tại chỉ hiển thị giả lập UI vì không có tác vụ)
        self.result_title.config(text="[Kết quả AI sẽ hiện ở đây]", fg="#3b82f6")
        self.result_desc.config(text="Giao diện Python đã hoàn thiện. Hãy chèn model AI của bạn vào hàm classify() trong file app.py này để nó hoạt động nhé!")
        
        # Khôi phục nút
        self.classify_btn.config(state=tk.NORMAL, text="Phân Loại Rác")

if __name__ == "__main__":
    root = tk.Tk()
    app = GarbageClassifierApp(root)
    root.mainloop()
