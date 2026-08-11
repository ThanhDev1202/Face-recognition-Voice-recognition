import os
import sys
import threading
import numpy as np
import cv2
import customtkinter as ctk
from PIL import Image
from tkinter import messagebox
import csv
from datetime import datetime

# Thêm thư mục gốc dự án (thư mục 'python') vào sys.path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# Import các module trong dự án
from Camera_Recorder.Camera import Camera
from detector.FaceDetector import FaceDetector
from recognizer.FaceRecognizer import FaceRecognizer


class FaceVerifyWindow(ctk.CTkToplevel):

    def __init__(self, parent, username):
        super().__init__(parent)

        self.result = False
        self.username = username
        self.is_running = True  # Cờ kiểm soát vòng lặp UI

        self.geometry("700x620")
        self.title(f"Xác thực khuôn mặt - {self.username}")
        self.resizable(False, False)

        # --------------------------------------------------
        # 1. Khởi tạo Models AI & Camera
        # --------------------------------------------------
        self.detector = FaceDetector(model_path="model/det_500m.onnx", conf_threshold=0.5)
        self.recognizer = FaceRecognizer(model_path="model/w600k_mbf.onnx")

        # Ngưỡng so sánh Cosine Similarity
        self.THRESHOLD = 0.80

        # Camera
        self.camera = Camera(camera_index=0, width=640, height=480)

        # --------------------------------------------------
        # 2. Giao diện (UI)
        # --------------------------------------------------
        self.lbl_title = ctk.CTkLabel(
            self,
            text=f"Đang xác thực cho tài khoản: {self.username}",
            font=("Arial", 16, "bold")
        )
        self.lbl_title.pack(pady=10)

        # Tối ưu CTkImage: Khởi tạo sẵn 1 đối tượng rỗng để tái sử dụng, tránh rò rỉ RAM
        placeholder_img = Image.new("RGB", (640, 480), color="black")
        self.ctk_img = ctk.CTkImage(light_image=placeholder_img, dark_image=placeholder_img, size=(640, 400))

        # Frame hiển thị Video Camera
        self.lbl_video = ctk.CTkLabel(
            self,
            text="",
            image=self.ctk_img,
            width=640,
            height=400,
            fg_color="black"
        )
        self.lbl_video.pack(pady=10)

        # Nút xác thực
        self.btn_verify = ctk.CTkButton(
            self,
            text="Xác nhận khuôn mặt",
            command=self.success,
            width=220,
            height=40,
            font=("Arial", 14, "bold")
        )
        self.btn_verify.pack(pady=15)

        # --------------------------------------------------
        # 3. Sự kiện & Vòng lặp Video
        # --------------------------------------------------
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        if self.camera.start():
            self.update_frame()
        else:
            self.lbl_video.configure(text="Không thể kết nối tới Webcam!", image="")

    def update_frame(self):
        """Render khung hình liên tục từ Webcam lên UI (Đã tối ưu RAM)"""
        if not self.is_running:
            return

        frame = self.camera.get_frame(copy=False)

        if frame is not None:
            frame = self.draw_oval(frame)
            # Chuyển BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Tái sử dụng đối tượng CTkImage sẵn có (KHÔNG tạo new CTkImage ở mỗi frame)
            self.ctk_img.configure(light_image=img, dark_image=img)

        # Lặp lại sau 30ms (~30 FPS) - Vừa đủ mượt và nhẹ CPU
        self.after(30, self.update_frame)

    def success(self):
        """Khởi chạy nhận diện AI trên luồng riêng (Thread)"""
        self.btn_verify.configure(state="disabled", text="Đang xử lý AI...")
        threading.Thread(target=self._process_verification, daemon=True).start()

    def _process_verification(self):
        """Luồng xử lý chính: Detect -> Extract -> Cosine Similarity"""
        # 1. Lấy khung hình hiện tại
        frame = self.camera.get_frame(copy=True)
        if frame is None:
            self._show_error("Không lấy được dữ liệu từ Camera!")
            return

        # 2. Phát hiện khuôn mặt bằng SCRFD
        bboxes, kpss = self.detector.detect(frame)

        if len(bboxes) == 0:
            self._show_error("Không tìm thấy khuôn mặt! Vui lòng nhìn thẳng vào camera.")
            return

        if len(bboxes) > 1:
            self._show_error("Phát hiện nhiều hơn 1 khuôn mặt! Vui lòng đứng một mình.")
            return

        # Lấy 5 điểm mốc (landmarks) của khuôn mặt duy nhất
        landmarks = kpss[0]

        # 3. Trích xuất Feature Embedding từ ảnh live
        embedding_live = self.recognizer.extract_embedding(frame, landmarks)
        # 4. Tải vector khuôn mặt mẫu trong cơ sở dữ liệu/file
        embedding_db = self.load_user_embedding_from_db(self.username)

        if embedding_db is None:
            self._show_error(f"Tài khoản '{self.username}' chưa đăng ký dữ liệu khuôn mặt!")
            return

        # 5. Tính Cosine Similarity
        score = self.recognizer.compute_cosine_similarity(
            embedding_live,
            embedding_db
        )

        # 6. So sánh với threshold
        if score >= self.THRESHOLD:
            result = "ACCEPT"
        else:
            result = "REJECT"

        # Ghi log
        self.log_verification(score, result)

        print(
            f"[Xác thực] User: {self.username} | "
            f"Score: {score:.4f} | "
            f"Threshold: {self.THRESHOLD:.4f} | "
            f"Result: {result}"
        )

        if result == "ACCEPT":
            self.result = True
            self.after(0, self.on_close)
        else:
            self._show_error(
                f"Xác thực thất bại!\n"
                f"Độ tương đồng: {score:.4f}"
            )

    def load_user_embedding_from_db(self, username):
        file_path = os.path.join(
            "database",
            "face_embeddings",
            f"{username}.npy"
        )

        if not os.path.exists(file_path):
            return None

        return np.load(file_path)

    def _show_error(self, message):
        """Hiển thị thông báo lỗi và mở lại nút bấm"""
        def _gui_update():
            messagebox.showerror("Thông báo", message, parent=self)
            if self.is_running:
                self.btn_verify.configure(state="normal", text="Xác nhận khuôn mặt")

        self.after(0, _gui_update)

    def draw_oval(self, frame):
        h, w = frame.shape[:2]
        # Vẽ oval trắng ở giữa màn hình
        cv2.ellipse(frame, (w // 2, h // 2), (int(w * 0.22), int(h * 0.38)), 0, 0, 360, (255, 255, 255), 2)
        return frame
    
    def on_close(self):
        """Dừng camera và đóng cửa sổ an toàn"""
        self.is_running = False  # Ngắt vòng lặp update_frame ngay lập tức
        self.camera.stop()
        self.destroy()

    def log_verification(self, score, result):
        os.makedirs("logs", exist_ok=True)

        log_file = os.path.join(
            "logs",
            "face_verification.csv"
        )

        file_exists = os.path.exists(log_file)

        with open(
            log_file,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "username",
                    "score",
                    "threshold",
                    "result"
                ])

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.username,
                f"{score:.6f}",
                f"{self.THRESHOLD:.6f}",
                result
            ])