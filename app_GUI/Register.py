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
from Camera_Recorder.Camera import Camera
from recognizer.FaceRecognizer import FaceRecognizer
from recognizer.VoiceRecognizer import VoiceRecognizer
from detector.FaceDetector import FaceDetector
from PreProcess.VoicePreProcess import VoicePreprocessor

# =========================================================
# GIAO DIỆN ĐĂNG KÝ MFA
# =========================================================
class RegisterWindow(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Đăng ký sinh trắc học MFA")
        self.geometry("850x500")
        self.resizable(False, False)

        # Sử dụng đối tượng Camera
        self.camera = Camera(camera_index=0, width=640, height=480)

        # Biến lưu trữ dữ liệu khuôn mặt & giọng nói
        self.face_embedding = None
        self.voice_embedding = None

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
        self.MAX_VOICE_SAMPLES = 3
        self.voice_guides = [
            "Lần 1/3: Nói một cách tự nhiên",
            "Lần 2/3: Nói lại câu vừa rồi với tốc độ bình thường",
            "Lần 3/3: Xác thực giọng nói lần cuối"
        ]

        # Khởi tạo AI Models & Preprocessor
        try:
            self.face_detector = FaceDetector()
            self.face_recognizer = FaceRecognizer(model_path="model/Face/w600k_mbf.onnx")
            self.voice_recognizer = VoiceRecognizer(model_dir="model/Voice")
            self.voice_preprocessor = VoicePreprocessor(sample_rate=16000)
        except Exception as e:
            messagebox.showerror("Lỗi Khởi Tạo AI", f"Không thể nạp model/preprocessor:\n{e}")

        # ==========================================
        # Bố cục Giao diện
        # ==========================================
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # --- CỘT TRÁI: CAMERA & KHUÔN MẶT ---
        left_frame = ctk.CTkFrame(self)
        left_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")

        ctk.CTkLabel(left_frame, text="1. Thu thập Khuôn mặt", font=("Arial", 16, "bold")).pack(pady=(10, 2))

        self.lbl_guide = ctk.CTkLabel(
            left_frame,
            text="Nhấn nút bên dưới để bắt đầu mở Camera",
            font=("Arial", 13, "bold"),
            text_color="#3498db"
        )
        self.lbl_guide.pack(pady=(0, 5))

        self.lbl_camera = ctk.CTkLabel(left_frame, text="Camera Off", width=360, height=270, fg_color="black")
        self.lbl_camera.pack(pady=5)

        face_btn_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        face_btn_frame.pack(fill="x", padx=10, pady=10)

        self.btn_capture_face = ctk.CTkButton(
            face_btn_frame,
            text="📸 Bắt đầu & Chụp mặt",
            command=self.capture_face
        )
        self.btn_capture_face.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.btn_reset_face = ctk.CTkButton(
            face_btn_frame,
            text="🔄 Chụp lại",
            width=90,
            fg_color="#e67e22",
            hover_color="#d35400",
            command=self.reset_face
        )
        self.btn_reset_face.pack(side="right")

        # --- CỘT PHẢI: GIỌNG NÓI & HOÀN TẤT ---
        right_frame = ctk.CTkFrame(self)
        right_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")

        ctk.CTkLabel(right_frame, text="2. Thu thập Giọng nói & Tài khoản", font=("Arial", 16, "bold")).pack(pady=10)

        ctk.CTkLabel(right_frame, text="Tên đăng nhập:", font=("Arial", 13)).pack(anchor="w", padx=30, pady=(10, 2))
        self.entry_user = ctk.CTkEntry(right_frame, width=300, placeholder_text="Nhập username...")
        self.entry_user.pack(pady=(0, 10))
        # Sự kiện kiểm tra khi người dùng rời khỏi ô nhập tên tài khoản
        self.entry_user.bind("<FocusOut>", self.check_username_exists)

        self.lbl_voice_status = ctk.CTkLabel(
            right_frame,
            text="Nhấn nút để ghi âm (3 giây)",
            font=("Arial", 12),
            text_color="gray70"
        )
        self.lbl_voice_status.pack(pady=5)

        self.progress_voice = ctk.CTkProgressBar(right_frame, width=300)
        self.progress_voice.set(0)
        self.progress_voice.pack(pady=5)

        voice_btn_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        voice_btn_frame.pack(fill="x", padx=25, pady=10)

        self.btn_record_voice = ctk.CTkButton(
            voice_btn_frame,
            text="🎤 Ghi âm mẫu (1/3)",
            command=self.record_voice,
            state="disabled",
            fg_color="gray30"
        )
        self.btn_record_voice.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.btn_reset_voice = ctk.CTkButton(
            voice_btn_frame,
            text="🔄 Ghi âm lại",
            width=90,
            fg_color="#e67e22",
            hover_color="#d35400",
            command=self.reset_voice
        )
        self.btn_reset_voice.pack(side="right")

        ctk.CTkFrame(right_frame, height=2, fg_color="gray50").pack(fill="x", padx=30, pady=10)

        self.btn_submit = ctk.CTkButton(
            right_frame,
            text="Hoàn tất Đăng ký",
            height=40,
            font=("Arial", 15, "bold"),
            command=self.submit_registration,
            state="disabled",
            fg_color="gray30"
        )
        self.btn_submit.pack(fill="x", padx=30, pady=5)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # #####################################################
    # KIỂM TRA TÀI KHOẢN TỒN TẠI
    # #####################################################

    def check_username_exists(self, event=None):
        """Kiểm tra xem tài khoản đã tồn tại trong cơ sở dữ liệu hay chưa"""
        username = self.entry_user.get().strip()
        if not username:
            return

        face_path = f"database/face_embeddings/{username}.npy"
        voice_path = f"database/voice_embeddings/{username}.npy"

        if os.path.exists(face_path) or os.path.exists(voice_path):
            self.lbl_guide.configure(
                text=f"⚠️ Tài khoản '{username}' đã tồn tại!", 
                text_color="#e74c3c"
            )
            messagebox.showerror("Lỗi", f"Tài khoản '{username}' đã tồn tại trong hệ thống! Vui lòng chọn tên khác.")
        else:
            if self.face_embedding is None:
                self.lbl_guide.configure(
                    text="Nhấn nút bên dưới để bắt đầu mở Camera", 
                    text_color="#3498db"
                )

    # #####################################################
    # CÁC HÀM RESET
    # #####################################################

    def reset_face(self):
        """Đặt lại toàn bộ quá trình chụp ảnh khuôn mặt"""
        self.stop_camera()
        self.face_embedding = None
        self.face_embeddings_list = []
        self.entry_user.configure(state="normal")

        self.lbl_guide.configure(
            text="Nhấn nút bên dưới để bắt đầu mở Camera",
            text_color="#3498db"
        )
        self.lbl_camera.configure(text="Camera Off", image="", fg_color="black")
        self.btn_capture_face.configure(
            text="📸 Bắt đầu & Chụp mặt",
            state="normal",
            fg_color=["#3a7ebf", "#1f538d"]
        )

        self.btn_submit.configure(state="disabled", fg_color="gray30")

    def reset_voice(self):
        """Đặt lại toàn bộ quá trình ghi âm giọng nói"""
        self.voice_embedding = None
        self.voice_embeddings_list = []

        self.lbl_voice_status.configure(
            text="Nhấn nút để ghi âm (3 giây)",
            text_color="gray70"
        )
        self.progress_voice.set(0)

        if self.face_embedding is not None:
            self.btn_record_voice.configure(
                text=f"🎤 Ghi âm mẫu (1/{self.MAX_VOICE_SAMPLES})",
                state="normal",
                fg_color=["#3a7ebf", "#1f538d"]
            )
        else:
            self.btn_record_voice.configure(
                text=f"🎤 Ghi âm mẫu (1/{self.MAX_VOICE_SAMPLES})",
                state="disabled",
                fg_color="gray30"
            )

        self.btn_submit.configure(state="disabled", fg_color="gray30")

    # #####################################################
    # TẬN DỤNG LỚP CAMERA ĐỂ CHỤP VÀ HIỂN THỊ
    # #####################################################

    def capture_face(self):
        username = self.entry_user.get().strip()
        if not username:
            messagebox.showwarning("Thông báo", "Vui lòng nhập Tên đăng nhập trước!")
            return

        # Kiểm tra nhanh trước khi cho phép mở camera
        face_path = f"database/face_embeddings/{username}.npy"
        voice_path = f"database/voice_embeddings/{username}.npy"
        if os.path.exists(face_path) or os.path.exists(voice_path):
            messagebox.showerror("Lỗi", f"Tài khoản '{username}' đã tồn tại! Không thể đăng ký trùng.")
            return

        self.entry_user.configure(state="disabled")

        # BƯỚC 1: Khởi tạo & Mở camera bằng lớp Camera
        if not self.camera.running:
            success = self.camera.start()
            if not success:
                messagebox.showerror("Lỗi", "Không thể mở Webcam!")
                self.entry_user.configure(state="normal")
                return

            self.face_embeddings_list = []
            self.lbl_guide.configure(text=f"📸 {self.face_guides[0]}", text_color="#e67e22")
            self.btn_capture_face.configure(text="📸 Chụp góc hiện tại", fg_color="#e67e22")

            self.update_camera_feed()

        # BƯỚC 2: Lấy frame mới nhất thông qua get_frame() để đưa sang Thread AI
        else:
            current_frame = self.camera.get_frame(copy=True)
            if current_frame is None:
                messagebox.showwarning("Cảnh báo", "Chưa nhận được tín hiệu hình ảnh từ camera!")
                return

            self.btn_capture_face.configure(state="disabled")

            # Xử lý lật ảnh cho giống kính soi (như lúc vẽ GUI)
            flipped_frame = cv2.flip(current_frame, 1)

            threading.Thread(
                target=self._process_face_thread,
                args=(flipped_frame,),
                daemon=True
            ).start()

    def _process_face_thread(self, frame):
        """Xử lý AI tách biệt hoàn toàn khỏi Main UI Thread"""
        bboxes, landmarks = self.face_detector.detect(frame)

        if len(landmarks) > 0:
            emb = self.face_recognizer.extract_embedding(frame, landmarks[0])
            emb = emb / np.linalg.norm(emb)
            self.face_embeddings_list.append(emb)

            count = len(self.face_embeddings_list)

            if count < self.MAX_FACE_SAMPLES:
                next_guide = self.face_guides[count]
                self.after(0, lambda: self.lbl_guide.configure(text=f"📸 {next_guide}", text_color="#e67e22"))
                self.after(0, lambda: self.btn_capture_face.configure(
                    text=f"📸 Chụp tiếp ({count}/{self.MAX_FACE_SAMPLES})",
                    state="normal"
                ))
            else:
                avg_emb = np.mean(self.face_embeddings_list, axis=0)
                self.face_embedding = avg_emb / np.linalg.norm(avg_emb)

                self.after(0, self._on_face_complete)
        else:
            self.after(0, lambda: messagebox.showwarning("Cảnh báo", "Không tìm thấy khuôn mặt trong khung hình! Hãy giữ nguyên tư thế và thử lại."))
            self.after(0, lambda: self.btn_capture_face.configure(state="normal"))

    def _on_face_complete(self):
        """Callback khi đã thu thập đủ 3 góc khuôn mặt"""
        self.stop_camera()
        self.lbl_guide.configure(text="Đã hoàn tất thu thập khuôn mặt!", text_color="#2ecc71")
        self.lbl_camera.configure(text=f"Đã lưu đủ {self.MAX_FACE_SAMPLES} góc mặt ✓", fg_color="#2b8a3e", image="")
        self.btn_capture_face.configure(text="Khuôn mặt đã lưu ✓", state="disabled", fg_color="#2b8a3e")

        self.btn_record_voice.configure(
            state="normal",
            fg_color=["#3a7ebf", "#1f538d"],
            text=f"🎤 Ghi âm mẫu (1/{self.MAX_VOICE_SAMPLES})"
        )
        messagebox.showinfo("Thành công", f"Đã trích xuất & tổng hợp đặc trưng từ {self.MAX_FACE_SAMPLES} góc mặt thành công!")

    def update_camera_feed(self):
        """Cập nhật giao diện Tkinter bằng cách lấy frame liên tục từ Camera thread"""
        if self.camera.running:
            frame = self.camera.get_frame(copy=True)
            if frame is not None:
                frame = cv2.flip(frame, 1)

                height, width, _ = frame.shape
                center_x, center_y = width // 2, height // 2
                axis_x, axis_y = 110, 150

                # Vẽ Oval hướng dẫn
                cv2.ellipse(frame, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, (0, 255, 255), 2)

                # Vẽ Text hướng dẫn
                current_step = len(self.face_embeddings_list)
                if current_step < self.MAX_FACE_SAMPLES:
                    guide_text = self.face_guides[current_step]
                    cv2.putText(
                        frame, guide_text, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA
                    )
                    cv2.putText(
                        frame, guide_text, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA
                    )

                rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(360, 270))

                self.lbl_camera.configure(image=ctk_img, text="")
                self.lbl_camera.image = ctk_img

            self.after(20, self.update_camera_feed)

    def stop_camera(self):
        """Gọi hàm stop() của Camera class"""
        self.camera.stop()

    # #####################################################
    # GHI ÂM ĐA MẪU
    # #####################################################

    def record_voice(self):
        self.btn_record_voice.configure(state="disabled")
        threading.Thread(target=self._record_voice_thread, daemon=True).start()

    def _record_voice_thread(self):
        duration = 3.0
        sample_rate = 16000
        current_step = len(self.voice_embeddings_list)

        guide_text = self.voice_guides[current_step] if current_step < len(self.voice_guides) else "Đang ghi âm..."
        self.after(0, lambda: self.lbl_voice_status.configure(
            text=f"🎙️ {guide_text}...", text_color="#e67e22"
        ))

        num_samples = int(duration * sample_rate)
        recording = sd.rec(num_samples, samplerate=sample_rate, channels=1, dtype="float32")

        start_time = time.time()
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            prog = min(elapsed / duration, 1.0)
            self.after(0, self._update_progress, prog)
            time.sleep(0.05)

        sd.wait()
        self.after(0, self._update_progress, 1.0)

        self.after(0, lambda: self.lbl_voice_status.configure(text="Đang tiền xử lý & phân tích...", text_color="#3498db"))

        raw_audio = recording.flatten()
        clean_audio = self.voice_preprocessor.process(raw_audio)

        if len(clean_audio) < int(sample_rate * 0.8):
            self.after(0, lambda: messagebox.showwarning("Cảnh báo", "Không phát hiện giọng nói hoặc âm thanh quá nhỏ! Hãy thử lại mẫu này."))
            self.after(0, lambda: self.lbl_voice_status.configure(text="Thử ghi âm lại lần này...", text_color="#e74c3c"))
            self.after(0, lambda: self.btn_record_voice.configure(state="normal"))
            return

        try:
            emb = self.voice_recognizer.extract_embedding_from_array(
                clean_audio,
                sample_rate=sample_rate
            )
            emb = emb / np.linalg.norm(emb)
            self.voice_embeddings_list.append(emb)

            self.after(0, self._on_voice_sample_success)

        except Exception as e:
            self.after(0, lambda e=e: messagebox.showerror("Lỗi Ghi Âm", f"Không thể xử lý giọng nói:\n{e}"))
            self.after(0, lambda: self.btn_record_voice.configure(state="normal"))

    def _update_progress(self, val):
        self.progress_voice.set(val)

    def _on_voice_sample_success(self):
        count = len(self.voice_embeddings_list)

        if count < self.MAX_VOICE_SAMPLES:
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
            avg_emb = np.mean(self.voice_embeddings_list, axis=0)
            self.voice_embedding = avg_emb / np.linalg.norm(avg_emb)

            self.lbl_voice_status.configure(
                text=f"Đã thu thập đủ {self.MAX_VOICE_SAMPLES} mẫu giọng nói ✓",
                text_color="#2ecc71"
            )
            self.btn_record_voice.configure(
                text="Giọng nói đã lưu ✓",
                state="disabled",
                fg_color="#2b8a3e"
            )

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
            os.makedirs("database/face_embeddings", exist_ok=True)
            os.makedirs("database/voice_embeddings", exist_ok=True)

            face_path = f"database/face_embeddings/{username}.npy"
            voice_path = f"database/voice_embeddings/{username}.npy"

            # Chặn hoàn toàn không cho ghi đè nếu đã tồn tại
            if os.path.exists(face_path) or os.path.exists(voice_path):
                messagebox.showerror("Lỗi Đăng Ký", f"Tài khoản '{username}' đã tồn tại trong hệ thống. Vui lòng sử dụng tên khác!")
                return

            np.save(face_path, self.face_embedding)
            np.save(voice_path, self.voice_embedding)

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