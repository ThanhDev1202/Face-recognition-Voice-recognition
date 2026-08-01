import os
import sys
import time
import threading
import numpy as np
import customtkinter as ctk
import sounddevice as sd
from tkinter import messagebox

# Thêm thư mục gốc dự án vào sys.path để import các module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recognizer.VoiceRecognizer import VoiceRecognizer
from PreProcess.VoicePreProcess import VoicePreprocessor


class VoiceVerifyWindow(ctk.CTkToplevel):

    def __init__(self, parent, username, sample_rate=16000, duration=3):
        super().__init__(parent)

        self.username = username
        self.sample_rate = sample_rate
        self.duration = duration  # Thời gian ghi âm (giây)
        self.result = False

        # Khởi tạo AI Recognizer & Preprocessor
        try:
            self.recognizer = VoiceRecognizer(model_dir="model/Voice")
            self.preprocessor = VoicePreprocessor(sample_rate=self.sample_rate)
        except Exception as e:
            messagebox.showerror("Lỗi Khởi Tạo", f"Không thể tải model hoặc preprocessor giọng nói:\n{e}")
            self.destroy()
            return

        # Cấu hình cửa sổ
        self.title("Xác thực giọng nói")
        self.geometry("450x320")
        self.resizable(False, False)

        # Modal Dialog Setup
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # UI Components
        ctk.CTkLabel(
            self,
            text=f"Xác thực giọng nói: {self.username}",
            font=("Arial", 16, "bold")
        ).pack(pady=(25, 10))

        self.lbl_status = ctk.CTkLabel(
            self,
            text="Nhấn nút và đọc câu thoại xác thực",
            font=("Arial", 13),
            text_color="gray70"
        )
        self.lbl_status.pack(pady=10)

        self.progress_bar = ctk.CTkProgressBar(self, width=280)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=15)

        self.btn_record = ctk.CTkButton(
            self,
            text="🎤 Bắt đầu ghi âm",
            width=200,
            command=self.start_verification
        )
        self.btn_record.pack(pady=15)

    def _update_ui(self, func, *args, **kwargs):
        """Hàm bổ trợ giúp thực thi việc cập nhật UI an toàn trên Main Thread"""
        self.after(0, lambda: func(*args, **kwargs))

    def start_verification(self):
        """Khởi chạy luồng ghi âm & xử lý AI riêng"""
        self.btn_record.configure(state="disabled")
        threading.Thread(target=self._process_verification_thread, daemon=True).start()

    def _process_verification_thread(self):
        try:
            # 1. Thông báo bắt đầu ghi âm
            self._update_ui(self.lbl_status.configure, text="Đang ghi âm, hãy nói...", text_color="#e67e22")
            
            num_samples = int(self.duration * self.sample_rate)
            
            # Bắt đầu ghi âm bất đồng bộ
            recording = sd.rec(
                num_samples, 
                samplerate=self.sample_rate, 
                channels=1, 
                dtype="float32"
            )
            
            # Cập nhật thanh tiến trình theo thời gian thực
            start_time = time.time()
            while time.time() - start_time < self.duration:
                elapsed = time.time() - start_time
                progress = elapsed / self.duration
                self._update_ui(self.progress_bar.set, progress)
                time.sleep(0.04)
            
            sd.wait()  # Chờ thu âm hoàn tất
            self._update_ui(self.progress_bar.set, 1.0)

            # 2. Tiền xử lý âm thanh (Bandpass Filter, Trim Silence, Normalize)
            self._update_ui(self.lbl_status.configure, text="Đang tiền xử lý & làm sạch âm thanh...", text_color="#3498db")
            
            raw_audio = recording.flatten()
            clean_audio = self.preprocessor.process(raw_audio)

            # Kiểm tra xem có giọng nói thực sự không
            if len(clean_audio) < int(self.sample_rate * 0.8):
                self._update_ui(
                    self.lbl_status.configure, 
                    text="Không phát hiện giọng nói hoặc âm thanh quá nhỏ!", 
                    text_color="#e74c3c"
                )
                self._update_ui(self.btn_record.configure, state="normal")
                return

            # 3. Trích xuất Feature Embedding từ âm thanh đã xử lý
            self._update_ui(self.lbl_status.configure, text="Đang phân tích đặc trưng giọng nói...", text_color="#3498db")
            
            live_embedding = self.recognizer.extract_embedding_from_array(
                clean_audio, 
                sample_rate=self.sample_rate
            )

            # 4. Lấy Embedding mẫu đã đăng ký từ DB/File
            enrolled_embedding = self._get_user_enrolled_embedding(self.username)

            if enrolled_embedding is None:
                self._update_ui(self.lbl_status.configure, text="Không tìm thấy dữ liệu giọng nói người dùng!", text_color="#e74c3c")
                self._update_ui(self.btn_record.configure, state="normal")
                return

            # 5. Tính Cosine Similarity & Đánh giá
            similarity = self.recognizer.compute_cosine_similarity(live_embedding, enrolled_embedding)
            THRESHOLD = 0.68 
            print(f"[Xác thực] User: {self.username} | Similarity: {similarity:.4f} | Threshold: {THRESHOLD:.2f}")

            if similarity >= THRESHOLD:
                self.result = True
                self._update_ui(
                    self.lbl_status.configure, 
                    text=f"Xác thực thành công! (Khớp: {similarity*100:.1f}%)", 
                    text_color="#2ecc71"
                )
                time.sleep(1.2)
                self._update_ui(self._close_window)
            else:
                self.result = False
                self._update_ui(
                    self.lbl_status.configure, 
                    text=f"Xác thực thất bại! (Độ tương đồng: {similarity*100:.1f}%)", 
                    text_color="#e74c3c"
                )
                self._update_ui(self.btn_record.configure, state="normal")

        except Exception as err:
            self._update_ui(self.lbl_status.configure, text=f"Lỗi hệ thống: {err}", text_color="#e74c3c")
            self._update_ui(self.btn_record.configure, state="normal")

    def _get_user_enrolled_embedding(self, username):
        """Lấy vector embedding đã lưu từ trước của user"""
        file_path = f"database/voice_embeddings/{username}.npy"
        if os.path.exists(file_path):
            return np.load(file_path)
        return None

    def _on_close(self):
        """Sự kiện click nút X đóng cửa sổ"""
        self.result = False
        self._close_window()

    def _close_window(self):
        self.grab_release()
        self.destroy()