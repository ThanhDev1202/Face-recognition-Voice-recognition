import os
import sys
import time
import threading
import cv2
import numpy as np
import customtkinter as ctk
import sounddevice as sd
from PIL import Image
from tkinter import messagebox

# Thêm thư mục gốc dự án vào sys.path để import các module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recognizer.FaceRecognizer import FaceRecognizer
from recognizer.VoiceRecognizer import VoiceRecognizer
from detector.FaceDetector import FaceDetector
from PreProcess.VoicePreProcess import VoicePreprocessor


class RegisterWindow(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Đăng ký sinh trắc học MFA")
        self.geometry("850x480")
        self.resizable(False, False)

        # Biến lưu trữ dữ liệu khuôn mặt & giọng nói
        self.face_embedding = None
        self.voice_embedding = None
        self.cap = None
        self.is_camera_running = False

        # --- Cấu hình thu thập đa góc mặt ---
        self.face_embeddings_list = []
        self.face_guides = [
            "1/3: Nhìn THẲNG vào camera",
            "2/3: Nghiêng mặt sang TRÁI",
            "3/3: Nghiêng mặt sang PHẢI"
        ]
        self.MAX_FACE_SAMPLES = len(self.face_guides)

        # --- Cấu hình thu thập đa mẫu giọng nói ---
        self.voice_embeddings_list = []
        self.MAX_VOICE_SAMPLES = 3  # Số lần ghi âm mong muốn
        self.voice_guides = [
            "Lần 1/3: Nói một cách tự nhiên",
            "Lần 2/3: Nói lại câu vừa rồi với tốc độ bình thường",
            "Lần 3/3: Xác thực giọng nói lần cuối"
        ]

        # Khởi tạo AI Models & Preprocessor
        try:
            self.face_detector = FaceDetector()  # Detector lấy 5 landmarks
            self.face_recognizer = FaceRecognizer(model_path="model/w600k_mbf.onnx")
            self.voice_recognizer = VoiceRecognizer(model_dir="model/Voice")
            self.voice_preprocessor = VoicePreprocessor(sample_rate=16000)
        except Exception as e:
            messagebox.showerror("Lỗi Khởi Tạo AI", f"Không thể nạp model/preprocessor:\n{e}")

        # ==========================================
        # Bố cục Giao diện (2 Cột Left/Right)
        # ==========================================
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # --- CỘT TRÁI: CAMERA & KHUÔN MẶT ---
        left_frame = ctk.CTkFrame(self)
        left_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")

        ctk.CTkLabel(left_frame, text="1. Thu thập Khuôn mặt", font=("Arial", 16, "bold")).pack(pady=(10, 2))

        # Label hướng dẫn người dùng góc chụp
        self.lbl_guide = ctk.CTkLabel(
            left_frame,
            text="Nhấn nút bên dưới để bắt đầu mở Camera",
            font=("Arial", 13, "bold"),
            text_color="#3498db"
        )
        self.lbl_guide.pack(pady=(0, 5))

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
            text="🎤 Ghi âm giọng nói (Lần 1/3)",
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
    # TRỰC TIẾP LẤY EMBEDDING KHUÔN MẶT (ĐA GÓC CHỤP)
    # #####################################################

    def capture_face(self):
        username = self.entry_user.get().strip()
        if not username:
            messagebox.showwarning("Thông báo", "Vui lòng nhập Tên đăng nhập trước!")
            return

        # Khóa Username để không sửa giữa chừng
        self.entry_user.configure(state="disabled")

        # --- BƯỚC 1: KHỞI TẠO CAMERA ---
        if not self.is_camera_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Lỗi", "Không thể mở Webcam!")
                return
            
            self.is_camera_running = True
            self.face_embeddings_list = []  # Reset danh sách vector
            
            # Cập nhật GUI cho mẫu đầu tiên
            self.lbl_guide.configure(text=f"📸 {self.face_guides[0]}", text_color="#e67e22")
            self.btn_capture_face.configure(text="📸 Chụp góc hiện tại", fg_color="#e67e22")
            self.update_camera_feed()

        # --- BƯỚC 2: THU THẬP TỪNG GÓC MẶT ---
        else:
            ret, frame = self.cap.read()
            if ret:
                # Đảm bảo đồng bộ lật gương (Mirror) giữa camera feed và frame xử lý
                frame = cv2.flip(frame, 1)

                # 1. Phát hiện khuôn mặt & Lấy 5 landmarks
                bboxes, landmarks = self.face_detector.detect(frame)

                if len(landmarks) > 0:
                    # Trích xuất & Chuẩn hóa L2 cho vector thành phần
                    emb = self.face_recognizer.extract_embedding(frame, landmarks[0])
                    emb = emb / np.linalg.norm(emb)
                    self.face_embeddings_list.append(emb)

                    count = len(self.face_embeddings_list)

                    # Chưa đủ 3 góc mặt
                    if count < self.MAX_FACE_SAMPLES:
                        next_guide = self.face_guides[count]
                        self.lbl_guide.configure(text=f"📸 {next_guide}", text_color="#e67e22")
                        self.btn_capture_face.configure(text=f"📸 Chụp tiếp ({count}/{self.MAX_FACE_SAMPLES})")
                    
                    # Đã gom đủ 3 mẫu
                    else:
                        # Tính Centroid Vector (Trung bình cộng) & Chuẩn hóa L2 lần cuối
                        avg_emb = np.mean(self.face_embeddings_list, axis=0)
                        self.face_embedding = avg_emb / np.linalg.norm(avg_emb)

                        # Dừng camera & Cập nhật UI Hoàn tất
                        self.stop_camera()
                        self.lbl_guide.configure(text="Đã hoàn tất thu thập khuôn mặt!", text_color="#2ecc71")
                        self.lbl_camera.configure(text=f"Đã lưu đủ {self.MAX_FACE_SAMPLES} góc mặt ✓", fg_color="#2b8a3e", image="")
                        self.btn_capture_face.configure(text="Khuôn mặt đã lưu ✓", state="disabled", fg_color="#2b8a3e")

                        # Kích hoạt bước tiếp theo (Ghi âm)
                        self.btn_record_voice.configure(
                            state="normal",
                            fg_color=["#3a7ebf", "#1f538d"],
                            text=f"🎤 Ghi âm mẫu (1/{self.MAX_VOICE_SAMPLES})"
                        )
                        messagebox.showinfo("Thành công", f"Đã trích xuất & tổng hợp đặc trưng từ {self.MAX_FACE_SAMPLES} góc mặt thành công!")
                else:
                    messagebox.showwarning("Cảnh báo", "Không tìm thấy khuôn mặt trong khung hình! Hãy giữ nguyên tư thế và thử lại.")

    def update_camera_feed(self):
        if self.is_camera_running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Lật gương webcam
                frame = cv2.flip(frame, 1)

                # Lấy kích thước khung hình camera
                height, width, _ = frame.shape
                center_x, center_y = width // 2, height // 2

                # Kích thước của khung Oval (Bán trục ngang a = 110, Bán trục dọc b = 150)
                axis_x, axis_y = 110, 150

                # 1. Vẽ khung Oval hướng dẫn
                cv2.ellipse(frame, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, (0, 255, 255), 2) # Color: Yellow (BGR)

                # 2. Vẽ dòng text hướng dẫn trực tiếp lên luồng Video
                current_step = len(self.face_embeddings_list)
                if current_step < self.MAX_FACE_SAMPLES:
                    guide_text = self.face_guides[current_step]
                    # Vẽ chữ có viền đen phía sau để dễ đọc
                    cv2.putText(
                        frame, guide_text, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA
                    )
                    cv2.putText(
                        frame, guide_text, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA
                    )

                # Chuyển BGR -> RGB hiển thị lên Tkinter
                rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(360, 270))
                
                self.lbl_camera.configure(image=ctk_img, text="")
                self.lbl_camera.image = ctk_img  # Đảm bảo giữ reference tránh Garbage Collector

            self.after(20, self.update_camera_feed)

    def stop_camera(self):
        self.is_camera_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # #####################################################
    # GHI ÂM ĐA MẪU & TIỀN XỬ LÝ ÂM THANH (THREAD-SAFE)
    # #####################################################

    def record_voice(self):
        self.btn_record_voice.configure(state="disabled")
        threading.Thread(target=self._record_voice_thread, daemon=True).start()

    def _record_voice_thread(self):
        duration = 3.0
        sample_rate = 16000
        current_step = len(self.voice_embeddings_list)

        # Cập nhật GUI thông báo bước ghi âm hiện tại
        guide_text = self.voice_guides[current_step] if current_step < len(self.voice_guides) else "Đang ghi âm..."
        self.after(0, lambda: self.lbl_voice_status.configure(
            text=f"🎙️ {guide_text}...", text_color="#e67e22"
        ))

        # 1. Ghi âm từ microphone
        num_samples = int(duration * sample_rate)
        recording = sd.rec(num_samples, samplerate=sample_rate, channels=1, dtype="float32")

        start_time = time.time()
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            prog = elapsed / duration
            self.after(0, lambda p=prog: self.progress_voice.set(p))
            time.sleep(0.05)

        sd.wait()
        self.after(0, lambda: self.progress_voice.set(1.0))

        # Cập nhật trạng thái xử lý
        self.after(0, lambda: self.lbl_voice_status.configure(text="Đang tiền xử lý & phân tích...", text_color="#3498db"))

        # 2. Tiền xử lý tín hiệu
        raw_audio = recording.flatten()
        clean_audio = self.voice_preprocessor.process(raw_audio)

        # Kiểm tra độ dài âm thanh sau khi VAD/cắt khoảng lặng
        if len(clean_audio) < int(sample_rate * 0.8):
            self.after(0, lambda: messagebox.showwarning("Cảnh báo", "Không phát hiện giọng nói hoặc âm thanh quá nhỏ! Hãy thử lại mẫu này."))
            self.after(0, lambda: self.lbl_voice_status.configure(text="Thử ghi âm lại lần này...", text_color="#e74c3c"))
            self.after(0, lambda: self.btn_record_voice.configure(state="normal"))
            return

        # 3. Trích xuất Embedding & Chuẩn hóa L2 cho mẫu hiện tại
        try:
            emb = self.voice_recognizer.extract_embedding_from_array(
                clean_audio,
                sample_rate=sample_rate
            )
            # Chuẩn hóa L2 từng mẫu riêng lẻ
            emb = emb / np.linalg.norm(emb)
            self.voice_embeddings_list.append(emb)

            # Chuyển về Main Thread để cập nhật UI
            self.after(0, self._on_voice_sample_success)

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Lỗi Ghi Âm", f"Không thể xử lý giọng nói:\n{e}"))
            self.after(0, lambda: self.btn_record_voice.configure(state="normal"))

    def _on_voice_sample_success(self):
        count = len(self.voice_embeddings_list)

        # Kiểm tra xem đã đủ số mẫu chưa
        if count < self.MAX_VOICE_SAMPLES:
            # Chưa đủ -> Cho phép bấm nút ghi âm mẫu tiếp theo
            self.lbl_voice_status.configure(
                text=f"Đã lưu {count}/{self.MAX_VOICE_SAMPLES} mẫu giọng nói ✓. Nhấn nút để tiếp tục.",
                text_color="#3498db"
            )
            self.btn_record_voice.configure(
                text=f"🎤 Ghi âm mẫu ({count + 1}/{self.MAX_VOICE_SAMPLES})",
                state="normal"
            )
            self.progress_voice.set(0)
        else:
            # Đã đủ mẫu -> Tính Vector Trung bình (Centroid) & L2 Normalize lần cuối
            avg_emb = np.mean(self.voice_embeddings_list, axis=0)
            self.voice_embedding = avg_emb / np.linalg.norm(avg_emb)

            # Cập nhật UI Hoàn tất
            self.lbl_voice_status.configure(
                text=f"Đã thu thập đủ {self.MAX_VOICE_SAMPLES} mẫu giọng nói ✓",
                text_color="#2ecc71"
            )
            self.btn_record_voice.configure(
                text="Giọng nói đã lưu ✓",
                state="disabled",
                fg_color="#2b8a3e"
            )

            # Kích hoạt nút Hoàn tất Đăng ký
            self.btn_submit.configure(state="normal", fg_color="#2b8a3e")
            messagebox.showinfo("Thành công", f"Đã trích xuất & tổng hợp đặc trưng từ {self.MAX_VOICE_SAMPLES} lần ghi âm thành công!")

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

            # Kiểm tra xem user đã tồn tại chưa
            face_path = f"database/face_embeddings/{username}.npy"
            if os.path.exists(face_path):
                if not messagebox.askyesno("Xác nhận", f"Tài khoản '{username}' đã tồn tại. Bạn có muốn GHI ĐÈ không?"):
                    return

            # Lưu vector dạng .npy
            np.save(face_path, self.face_embedding)
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