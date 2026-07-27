import os
import sys
import time
import threading
import cv2
import numpy as np
import customtkinter as ctk
import sounddevice as sd
from PIL import Image, ImageTk
from tkinter import messagebox

# Thêm thư mục gốc dự án vào sys.path để import recognizer & detector
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recognizer.FaceRecognizer import FaceRecognizer
from recognizer.VoiceRecognizer import VoiceRecognizer
from detector.FaceDetector import FaceDetector  # Giả định bạn có FaceDetector


class RegisterWindow(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Đăng ký sinh trắc học MFA")
        self.geometry("850x550")
        self.resizable(False, False)

        # Biến lưu trữ dữ liệu
        self.face_embedding = None
        self.voice_embedding = None
        self.cap = None
        self.is_camera_running = False

        # Khởi tạo AI Recognition Models
        try:
            self.face_detector = FaceDetector()  # Detector dùng để lấy 5 landmarks
            self.face_recognizer = FaceRecognizer(model_path="model/w600k_mbf.onnx")
            self.voice_recognizer = VoiceRecognizer(model_dir="model/Voice")
        except Exception as e:
            messagebox.showerror("Lỗi Khởi Tạo AI", f"Không thể nạp model AI:\n{e}")

        # ==========================================
        # Bố cục Giao diện (2 Cột Left/Right)
        # ==========================================
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # --- CỘT TRÁI: CAMERA & KHUÔN MẶT ---
        left_frame = ctk.CTkFrame(self)
        left_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")

        ctk.CTkLabel(left_frame, text="1. Thu thập Khuôn mặt", font=("Arial", 16, "bold")).pack(pady=10)

        # Màn hình hiển thị Camera
        self.lbl_camera = ctk.CTkLabel(left_frame, text="Camera Off", width=360, height=270, fg_color="black")
        self.lbl_camera.pack(pady=5)

        self.btn_capture_face = ctk.CTkButton(
            left_frame,
            text="📸 Bắt đầu Camera & Chụp mặt",
            command=self.capture_face
        )
        self.btn_capture_face.pack(pady=15)

        # --- CỘT PHẢI: GIỌNG NÓI & HOÀN TẤT ---
        right_frame = ctk.CTkFrame(self)
        right_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")

        ctk.CTkLabel(right_frame, text="2. Thu thập Giọng nói & Tài khoản", font=("Arial", 16, "bold")).pack(pady=10)

        # Username Input
        ctk.CTkLabel(right_frame, text="Tên đăng nhập:", font=("Arial", 13)).pack(anchor="w", padx=30, pady=(10, 2))
        self.entry_user = ctk.CTkEntry(right_frame, width=300, placeholder_text="Nhập username...")
        self.entry_user.pack(pady=(0, 15))

        # Status & Progress Bar Ghi âm
        self.lbl_voice_status = ctk.CTkLabel(
            right_frame,
            text="Nhấn nút để ghi âm (3 giây)",
            font=("Arial", 12),
            text_color="gray70"
        )
        self.lbl_voice_status.pack(pady=5)

        self.progress_voice = ctk.CTkProgressBar(right_frame, width=300)
        self.progress_voice.set(0)
        self.progress_voice.pack(pady=10)

        self.btn_record_voice = ctk.CTkButton(
            right_frame,
            text="🎤 Ghi âm giọng nói",
            command=self.record_voice,
            state="disabled",
            fg_color="gray30"
        )
        self.btn_record_voice.pack(pady=15)

        # Separator Line
        ctk.CTkFrame(right_frame, height=2, fg_color="gray50").pack(fill="x", padx=30, pady=15)

        # Nút Hoàn tất
        self.btn_submit = ctk.CTkButton(
            right_frame,
            text="Hoàn tất Đăng ký",
            height=40,
            font=("Arial", 15, "bold"),
            command=self.submit_registration,
            state="disabled",
            fg_color="gray30"
        )
        self.btn_submit.pack(pady=10)

        # Sự kiện đóng cửa sổ
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # #####################################################
    # TRỰC TIẾP LẤY EMBEDDING KHUÔN MẶT
    # #####################################################

    def capture_face(self):
        username = self.entry_user.get().strip()
        if not username:
            messagebox.showwarning("Thông báo", "Vui lòng nhập Tên đăng nhập trước!")
            return

        # Khóa Username để không sửa giữa chừng
        self.entry_user.configure(state="disabled")

        if not self.is_camera_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Lỗi", "Không thể mở Webcam!")
                return
            self.is_camera_running = True
            self.btn_capture_face.configure(text="📸 Chụp & Trích xuất", fg_color="#e67e22")
            self.update_camera_feed()
        else:
            # Khi người dùng bấm nút chụp
            ret, frame = self.cap.read()
            if ret:
                # 1. Phát hiện khuôn mặt & Lấy 5 landmarks
                bboxes, landmarks = self.face_detector.detect(frame)

                if len(landmarks) > 0:
                    # 2. Dùng FaceRecognizer trích xuất embedding trực tiếp
                    self.face_embedding = self.face_recognizer.extract_embedding(frame, landmarks[0])

                    # Dừng camera
                    self.stop_camera()
                    self.lbl_camera.configure(text="Đã chụp khuôn mặt ✓", fg_color="#2b8a3e")
                    self.btn_capture_face.configure(text="Khuôn mặt đã lưu ✓", state="disabled", fg_color="#2b8a3e")

                    # Kích hoạt nút ghi âm
                    self.btn_record_voice.configure(state="normal", fg_color=["#3a7ebf", "#1f538d"])
                    messagebox.showinfo("Thành công", "Trích xuất đặc trưng khuôn mặt thành công!")
                else:
                    messagebox.showwarning("Cảnh báo", "Không tìm thấy khuôn mặt trong khung hình! Hãy thử lại.")

    def update_camera_feed(self):
        if self.is_camera_running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Chuyển BGR -> RGB hiển thị lên Tkinter
                rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(360, 270))
                self.lbl_camera.configure(image=ctk_img, text="")
                self.lbl_camera.image = ctk_img

            self.after(20, self.update_camera_feed)

    def stop_camera(self):
        self.is_camera_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # #####################################################
    # TRỰC TIẾP LẤY EMBEDDING GIỌNG NÓI
    # #####################################################

    def record_voice(self):
        self.btn_record_voice.configure(state="disabled")
        threading.Thread(target=self._record_voice_thread, daemon=True).start()

    def _record_voice_thread(self):
        duration = 3.0
        sample_rate = 16000

        self.lbl_voice_status.configure(text="Đang ghi âm, hãy nói...", text_color="#e67e22")

        # 1. Thu âm từ Micro
        num_samples = int(duration * sample_rate)
        recording = sd.rec(num_samples, samplerate=sample_rate, channels=1, dtype="float32")

        start_time = time.time()
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            self.progress_voice.set(elapsed / duration)
            time.sleep(0.05)

        sd.wait()
        self.progress_voice.set(1.0)

        # 2. Dùng VoiceRecognizer trích xuất embedding trực tiếp
        self.lbl_voice_status.configure(text="Đang phân tích giọng nói...", text_color="#3498db")
        audio_data = recording.flatten()

        self.voice_embedding = self.voice_recognizer.extract_embedding_from_array(
            audio_data,
            sample_rate=sample_rate
        )

        # Cập nhật UI sau khi hoàn tất
        self.lbl_voice_status.configure(text="Đã ghi âm giọng nói ✓", text_color="#2ecc71")
        self.btn_record_voice.configure(text="Giọng nói đã lưu ✓", state="disabled", fg_color="#2b8a3e")

        # Kích hoạt nút Submit Hoàn tất
        self.btn_submit.configure(state="normal", fg_color="#2b8a3e")
        messagebox.showinfo("Thành công", "Trích xuất đặc trưng giọng nói thành công!")

    # #####################################################
    # HOÀN TẤT VÀ LƯU DATABASE
    # #####################################################

    def submit_registration(self):
        username = self.entry_user.get().strip()

        if self.face_embedding is None or self.voice_embedding is None:
            messagebox.showerror("Lỗi", "Chưa thu thập đủ dữ liệu Khuôn mặt và Giọng nói.")
            return

        try:
            # Tạo thư mục lưu trữ nếu chưa có
            os.makedirs("database/face_embeddings", exist_ok=True)
            os.makedirs("database/voice_embeddings", exist_ok=True)

            # Lưu vector dạng .npy
            np.save(f"database/face_embeddings/{username}.npy", self.face_embedding)
            np.save(f"database/voice_embeddings/{username}.npy", self.voice_embedding)

            messagebox.showinfo("Thành công", f"Đã đăng ký thành công cho tài khoản: {username}")
            self.on_close()

        except Exception as err:
            messagebox.showerror("Lỗi Lưu Dữ Liệu", f"Không thể lưu file .npy:\n{err}")

    def on_close(self):
        self.stop_camera()
        self.destroy()


if __name__ == "__main__":
    app = RegisterWindow()
    app.mainloop()