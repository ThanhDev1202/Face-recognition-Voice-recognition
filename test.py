import os
import cv2
import time
import customtkinter as ctk
from PIL import Image

# ==========================
# Cấu hình giao diện & Model
# ==========================
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

# Tải bộ phát hiện khuôn mặt Haar Cascade của OpenCV
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

POSES = [
    "1. Nhìn THẲNG vào camera",
    "2. Nghiêng đầu sang TRÁI",
    "3. Nghiêng đầu sang PHẢI",
    "4. Hơi CÚI ĐẦU xuống",
    "5. Hơi NGỬA ĐẦU lên",
]


class RegisterGUI(ctk.CTk):

    def __init__(self):
        super().__init__()

        # Cấu hình cửa sổ
        self.title("Đăng ký khuôn mặt đa góc độ")
        self.geometry("800x670")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.close_app)

        # Biến trạng thái
        self.cap = None
        self.running = False
        self.current_frame = None
        self.detected_faces = []   # Lưu danh sách tọa độ khuôn mặt phát hiện được
        self.current_pose_idx = 0
        self.flash_until = 0

        # Tiêu đề
        title = ctk.CTkLabel(
            self,
            text="ĐĂNG KÝ KHUÔN MẶT ĐA GÓC ĐỘ",
            font=("Arial", 24, "bold")
        )
        title.pack(pady=10)

        # Khung Camera
        self.camera_label = ctk.CTkLabel(
            self,
            text="Camera chưa mở",
            width=640,
            height=400,
            fg_color="#DDDDDD",
            corner_radius=8
        )
        self.camera_label.pack(pady=5)

        # Form nhập tên
        form = ctk.CTkFrame(self)
        form.pack(pady=10)

        ctk.CTkLabel(
            form,
            text="Họ và tên:",
            font=("Arial", 15, "bold")
        ).grid(row=0, column=0, padx=10, pady=5)

        self.name_entry = ctk.CTkEntry(
            form,
            width=300,
            placeholder_text="Nhập họ và tên đầy đủ..."
        )
        self.name_entry.grid(row=0, column=1, padx=10, pady=5)

        # Nút chức năng
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(pady=5)

        self.btn_start = ctk.CTkButton(
            button_frame,
            text="Mở Camera",
            width=130,
            command=self.start_camera
        )
        self.btn_start.grid(row=0, column=0, padx=6)

        self.btn_register = ctk.CTkButton(
            button_frame,
            text="Chụp góc này",
            width=130,
            fg_color="green",
            hover_color="#006400",
            command=self.capture_face
        )
        self.btn_register.grid(row=0, column=1, padx=6)

        self.btn_reset = ctk.CTkButton(
            button_frame,
            text="Chụp lại từ đầu",
            width=130,
            fg_color="#E67E22",
            hover_color="#D35400",
            command=self.reset_process
        )
        self.btn_reset.grid(row=0, column=2, padx=6)

        self.btn_exit = ctk.CTkButton(
            button_frame,
            text="Thoát",
            width=110,
            fg_color="red",
            hover_color="#8B0000",
            command=self.close_app
        )
        self.btn_exit.grid(row=0, column=3, padx=6)

        # Trạng thái & Hướng dẫn góc
        self.instruction_label = ctk.CTkLabel(
            self,
            text=f"Hướng dẫn: {POSES[0]}",
            font=("Arial", 16, "bold"),
            text_color="#1F618D"
        )
        self.instruction_label.pack(pady=5)

        self.status = ctk.CTkLabel(
            self,
            text="Trạng thái: Sẵn sàng",
            font=("Arial", 13, "italic")
        )
        self.status.pack(pady=2)

    def start_camera(self):
        if self.running:
            return

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.status.configure(text="Không thể kết nối với Camera!", text_color="red")
            return

        self.running = True
        self.btn_start.configure(state="disabled")
        self.status.configure(text="Camera đang hoạt động", text_color="black")
        self.update_frame()

    def update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            # 1. Lưu ảnh gốc SẠCH (dùng để crop khuôn mặt khi lưu)
            self.current_frame = frame.copy()

            # 2. Thực hiện Face Detection trên ảnh xám (để tăng tốc độ)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100)
            )
            self.detected_faces = faces  # Lưu vị trí khuôn mặt hiện tại

            # 3. Vẽ hình Oval khung ngắm
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            axes = (100, 160)

            # Màu chớp khi chụp thành công
            if time.time() < self.flash_until:
                oval_color = (0, 255, 0)  # Green
                thickness = 4
            else:
                oval_color = (255, 255, 0)  # Cyan
                thickness = 2

            cv2.ellipse(frame, center, axes, 0, 0, 360, oval_color, thickness)

            # 4. Vẽ khung Bounding Box quanh các khuôn mặt tìm thấy
            for (x, y, fw, fh) in faces:
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 255), 2)
                cv2.putText(
                    frame, "Face Detected", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )

            # Đổi sang RGB để hiển thị trên GUI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            ctk_img = ctk.CTkImage(light_image=image, dark_image=image, size=(640, 400))
            self.camera_label.configure(image=ctk_img, text="")

        self.after(30, self.update_frame)

    def capture_face(self):
        if self.current_frame is None or not self.running:
            self.status.configure(text="Vui lòng mở Camera trước khi chụp!", text_color="red")
            return

        name = self.name_entry.get().strip()
        if not name:
            self.status.configure(text="Vui lòng nhập họ và tên!", text_color="red")
            return

        # Kiểm tra điều kiện Face Detection
        if len(self.detected_faces) == 0:
            self.status.configure(text="⚠️ Không tìm thấy khuôn mặt nào! Hãy đứng vào khung hình.", text_color="red")
            return
        elif len(self.detected_faces) > 1:
            self.status.configure(text="⚠️ Phát hiện nhiều hơn 1 khuôn mặt! Chỉ chụp 1 người.", text_color="red")
            return

        # Khóa ô nhập tên trong quá trình chụp
        self.name_entry.configure(state="disabled")

        # Lấy vị trí khuôn mặt duy nhất
        (x, y, w, h) = self.detected_faces[0]

        # Mở rộng biên (padding) thêm 20% quanh mặt để lấy trọn cằm và tóc
        img_h, img_w = self.current_frame.shape[:2]
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + w + pad_x)
        y2 = min(img_h, y + h + pad_y)

        # Cắt khuôn mặt từ ảnh gốc
        face_crop = self.current_frame[y1:y2, x1:x2]

        # Tạo thư mục lưu trữ
        forbidden_chars = r'\/:*?"<>|'
        clean_name = "".join([c for c in name if c not in forbidden_chars])
        user_dir = os.path.join("dataset", clean_name)
        os.makedirs(user_dir, exist_ok=True)

        filename = os.path.join(user_dir, f"goc_{self.current_pose_idx + 1}.jpg")

        # Lưu ảnh đã CROP
        is_success, buffer = cv2.imencode(".jpg", face_crop)
        if is_success:
            with open(filename, "wb") as f:
                f.write(buffer)

            # Tín hiệu chớp xanh lá
            self.flash_until = time.time() + 0.3

            # Chuyển góc tiếp theo
            self.current_pose_idx += 1

            if self.current_pose_idx < len(POSES):
                self.instruction_label.configure(
                    text=f"Hướng dẫn: {POSES[self.current_pose_idx]}",
                    text_color="#1F618D"
                )
                self.status.configure(
                    text=f"Đã chụp & cắt khuôn mặt góc {self.current_pose_idx}/{len(POSES)}!",
                    text_color="green"
                )
            else:
                self.instruction_label.configure(
                    text="🎉 ĐÃ HOÀN THÀNH TẤT CẢ CÁC GÓC!",
                    text_color="green"
                )
                self.status.configure(
                    text=f"Đã đăng ký đầy đủ {len(POSES)} góc cho '{clean_name}'!",
                    text_color="green"
                )
                self.btn_register.configure(state="disabled")

    def reset_process(self):
        self.current_pose_idx = 0
        self.name_entry.configure(state="normal")
        self.instruction_label.configure(
            text=f"Hướng dẫn: {POSES[0]}",
            text_color="#1F618D"
        )
        self.status.configure(text="Đã làm mới tiến trình chụp.", text_color="black")
        self.btn_register.configure(state="normal")

    def close_app(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.destroy()


if __name__ == "__main__":
    app = RegisterGUI()
    app.mainloop()