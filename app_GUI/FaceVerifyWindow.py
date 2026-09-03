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
import gc

# Thêm thư mục gốc dự án (thư mục 'python') vào sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Import các module trong dự án
from Camera.Camera import Camera
from detector.FaceDetector import FaceDetector
from recognizer.FaceRecognizer import FaceRecognizer
from detector.AntiSpoofing import AntiSpoofing


class FaceVerifyWindow(ctk.CTkToplevel):

    def __init__(self, parent, username):
        super().__init__(parent)

        # Đồng bộ Working Directory về thư mục gốc dự án để tránh lỗi relative path
        os.chdir(BASE_DIR)

        self.result = False
        self.username = username
        self.is_running = True      # Cờ kiểm soát vòng lặp UI
        self.is_processing = False  # Cờ chống spam click nút bấm

        self.geometry("700x620")
        self.title(f"Xác thực khuôn mặt - {self.username}")
        self.resizable(False, False)

        # --------------------------------------------------
        # 1. Khởi tạo Models AI & Camera
        # --------------------------------------------------
        self.detector = FaceDetector(model_path="model/Face/det_500m.onnx", conf_threshold=0.5)
        self.recognizer = FaceRecognizer(model_path="model/Face/w600k_mbf.onnx")
        
        # Sử dụng đúng tên file mô hình có sẵn trong thư mục
        self.anti_spoof = AntiSpoofing(model_path="model/Face/mobilenetv3_large.onnx")
        
        self.SPOOF_THRESHOLD = 0.7  # Ngưỡng xác thực mặt thật
        self.THRESHOLD = 0.68       # Ngưỡng Cosine Similarity

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

        # Tối ưu CTkImage: Khởi tạo sẵn 1 đối tượng rỗng
        placeholder_img = Image.new("RGB", (640, 400), color="black")
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
        """Render khung hình liên tục từ Webcam lên UI (Thu gom rác RAM)"""
        if not self.is_running:
            return

        frame = self.camera.get_frame(copy=False)

        if frame is not None:
            frame = self.draw_oval(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (640, 400), interpolation=cv2.INTER_NEAREST)
            
            img = Image.fromarray(frame_resized)
            self.ctk_img.configure(light_image=img, dark_image=img)
            img.close()  # Giải phóng bộ nhớ tạm thời của PIL

        self.after(33, self.update_frame)

    def success(self):
        """Khởi chạy nhận diện AI trên luồng riêng (Chống spam click)"""
        if self.is_processing:
            return

        self.is_processing = True
        self.btn_verify.configure(state="disabled", text="Đang xử lý AI...")
        threading.Thread(target=self._process_verification, daemon=True).start()

    def _process_verification(self):
        """Luồng xử lý chính: Detect -> Anti-Spoofing -> Extract -> Matching"""
        try:
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

            # Lấy bbox và landmarks của khuôn mặt duy nhất
            bbox = bboxes[0]
            landmarks = kpss[0]

            # --------------------------------------------------
            # 2.5 KIỂM TRA MẶT THẬT / GIẢ MẠO (ANTI-SPOOFING)
            # --------------------------------------------------
            real_score = self.anti_spoof.predict(frame, bbox)
            print(f"[Anti-Spoofing] Real Score: {real_score:.4f} | Threshold: {self.SPOOF_THRESHOLD}")

            if real_score < self.SPOOF_THRESHOLD:
                self._show_error(
                    f"Cảnh báo: Phát hiện khuôn mặt không hợp lệ (Giả mạo)!\n"
                    f"Độ tin cậy mặt thật: {real_score * 100:.1f}%"
                )
                self.log_verification(score=0.0, result="SPOOF_REJECT")
                return

            # 3. Trích xuất Feature Embedding từ ảnh live
            embedding_live = self.recognizer.extract_embedding(frame, landmarks)
            
            # Chuẩn hóa L2 cho vector live
            norm_live = np.linalg.norm(embedding_live)
            if norm_live > 0:
                embedding_live = embedding_live / norm_live

            # 4. Tải dữ liệu khuôn mặt từ database (.npy)
            embedding_db = self.load_user_embedding_from_db(self.username)

            if embedding_db is None:
                self._show_error(f"Tài khoản '{self.username}' chưa đăng ký dữ liệu khuôn mặt!")
                return

            # 5. Tính Cosine Similarity
            score = self._compute_matrix_similarity(embedding_live, embedding_db)

            # 6. So sánh với threshold
            result = "ACCEPT" if score >= self.THRESHOLD else "REJECT"

            # Ghi log
            self.log_verification(score, result)

            print(
                f"[Xác thực] User: {self.username} | "
                f"Max Score: {score:.4f} | "
                f"Threshold: {self.THRESHOLD:.4f} | "
                f"Result: {result}"
            )

            if result == "ACCEPT":
                self.result = True
                self.after(0, self.on_close)
            else:
                self._show_error(
                    f"Xác thực thất bại!\n"
                    f"Độ tương đồng cao nhất: {score:.4f}"
                )
        except Exception as e:
            print(f"[Error] Lỗi xử lý xác thực: {e}")
            self._show_error("Đã xảy ra lỗi trong quá trình xử lý AI!")

    def _compute_matrix_similarity(self, live_emb, db_emb):
        """Tính Cosine Similarity giữa vector live_emb và ma trận/vector db_emb."""
        live_emb = live_emb.flatten()

        if db_emb.ndim == 1:
            norm_db = np.linalg.norm(db_emb)
            if norm_db > 0:
                db_emb = db_emb / norm_db
            return float(np.dot(live_emb, db_emb))

        elif db_emb.ndim == 2:
            norms = np.linalg.norm(db_emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            db_emb_norm = db_emb / norms

            scores = np.dot(db_emb_norm, live_emb)
            return float(np.max(scores))

        return 0.0

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
            self.is_processing = False
            if self.is_running:
                self.btn_verify.configure(state="normal", text="Xác nhận khuôn mặt")

        self.after(0, _gui_update)

    def draw_oval(self, frame):
        h, w = frame.shape[:2]
        cv2.ellipse(frame, (w // 2, h // 2), (int(w * 0.22), int(h * 0.38)), 0, 0, 360, (255, 255, 255), 2)
        return frame

    def release_resources(self):
        """Giải phóng toàn bộ model AI và camera"""

        if hasattr(self, "camera") and self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
            self.camera = None

        if hasattr(self, "detector") and self.detector is not None:
            try:
                self.detector.session = None
            except Exception:
                pass
            self.detector = None

        if hasattr(self, "recognizer") and self.recognizer is not None:
            try:
                self.recognizer.session = None
            except Exception:
                pass
            self.recognizer = None

        if hasattr(self, "anti_spoof") and self.anti_spoof is not None:
            try:
                if hasattr(self.anti_spoof, "release"):
                    self.anti_spoof.release()
                else:
                    self.anti_spoof.session = None
            except Exception:
                pass
            self.anti_spoof = None

        gc.collect()
        print("[FaceVerify] Đã giải phóng Face Models + AntiSpoofing + Camera")

    def on_close(self):
        """Dừng camera, giải phóng AI models và đóng cửa sổ"""
        if not self.is_running:
            return

        self.is_running = False
        self.release_resources()
        self.destroy()

    def log_verification(self, score, result):
        os.makedirs("logs", exist_ok=True)
        log_file = os.path.join("logs", "face_verification.csv")
        file_exists = os.path.exists(log_file)

        try:
            with open(log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow(["timestamp", "username", "score", "threshold", "result"])

                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.username,
                    f"{score:.6f}",
                    f"{self.THRESHOLD:.6f}",
                    result
                ])
        except Exception as e:
            print(f"[Log Error] Không thể ghi log: {e}")