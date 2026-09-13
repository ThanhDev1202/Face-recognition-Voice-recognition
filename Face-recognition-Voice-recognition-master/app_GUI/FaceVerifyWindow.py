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
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Import các module trong dự án
from Camera.Camera import Camera
from detector.FaceDetector import FaceDetector
from detector.LivenessDetector import LivenessDetector
from recognizer.FaceRecognizer import FaceRecognizer
from detector.AntiSpoofing import AntiSpoofing


class FaceVerifyWindow(ctk.CTkToplevel):

    def __init__(self, parent, username):
        print(" FACE VERIFY FILE ĐANG CHẠY:")
        print(__file__)
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
        self.liveness = LivenessDetector(
             movement_threshold=0.04,
             vertical_threshold=0.03,
             tilt_threshold=10.0,
             smile_threshold=0.06,
             min_frames=10,
             timeout=12.0
        )
        self.liveness_running = False
        self.liveness_passed = False
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
     # Khi Liveness đang chạy,
     # _update_liveness() sẽ chịu trách nhiệm hiển thị camera.
        if self.liveness_running:
             self.after(33, self.update_frame)
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
         print("========== SUCCESS() CUA LIVENESS DANG CHAY ==========")
         """Bắt đầu kiểm tra Liveness trước khi xác thực khuôn mặt."""
         if self.is_processing:
             return

         self.is_processing = True
         self.liveness_running = True
         self.liveness_passed = False

         self.liveness.reset()
         self.liveness.start()
         print(
             "[LIVENESS] ================================="
         )
         print("[LIVENESS] Bắt đầu kiểm tra:")
         print(
             "[LIVENESS] Sequence:",self.liveness.get_sequence())
         print(
             "[LIVENESS] ================================="
         )
         self.btn_verify.configure(
             state="disabled",
             text="Đang kiểm tra người thật..."
         )

    # Chỉ bắt đầu Liveness
         self._update_liveness()

    def _process_verification(self):
         print("========== PROCESS VERIFICATION DANG CHAY ==========")
         try:
        # ==========================================
        # Kiểm tra Liveness
        # ==========================================

             if not self.liveness_passed:

                 self._show_error(
                     "Liveness chưa đạt."
                 )

                 return

        # ==========================================
        # Lấy frame cuối
        # ==========================================

             frame = self.camera.get_frame(
                  copy=True
             )

             if frame is None:

                 self._show_error(
                     "Không lấy được hình ảnh từ camera."
                 )

                 return

        # ==========================================
        # Face Detection
        # ==========================================

             bboxes, kpss = self.detector.detect(
                 frame
             )

             if len(bboxes) == 0:

                 self._show_error(
                     "Không phát hiện khuôn mặt."
                 )

                 return

             if len(bboxes) > 1:

                 self._show_error(
                     "Phát hiện nhiều khuôn mặt."
                 )

                 return

             bbox = bboxes[0]
             landmarks = kpss[0]

        # ==========================================
        # Anti-Spoofing
        # ==========================================

             real_score = self.anti_spoof.predict(
                 frame,
                 bbox
             )

             print(
                 f"[ANTI-SPOOF] "
                 f"Score: {real_score:.4f} "
                 f"Threshold: {self.SPOOF_THRESHOLD:.4f}"
             )

             if real_score < self.SPOOF_THRESHOLD:

                 self.log_verification(
                     real_score,
                     "SPOOF_REJECT"
                 )

                 self._show_error(
                     f"Phát hiện khả năng giả mạo.\n\n"
                     f"Anti-Spoof Score: {real_score:.4f}"
                 )

                 return

        # ==========================================
        # Face Embedding
        # ==========================================

             embedding_live = (
                 self.recognizer.extract_embedding(
                     frame,
                     landmarks
                 )
             )

             embedding_live = np.asarray(
                 embedding_live,
                 dtype=np.float32
             )

             norm = np.linalg.norm(
                 embedding_live
             )

             if norm == 0:

                 self._show_error(
                     "Không tạo được face embedding."
                 )

                 return

             embedding_live /= norm

        # ==========================================
        # Load database embedding
        # ==========================================

             embedding_db = (
                 self.load_user_embedding_from_db(
                     self.username
                 )
             )

             if embedding_db is None:

                 self._show_error(
                     "Không tìm thấy dữ liệu khuôn mặt."
                 )

                 return

        # ==========================================
        # Face Similarity
        # ==========================================

             score = self._compute_matrix_similarity(
                 embedding_live,
                 embedding_db
             )

             print(
                 f"[FACE] "
                 f"Score: {score:.4f} "
                 f"Threshold: {self.THRESHOLD:.4f}"
             )

        # ==========================================
        # FACE ACCEPT
        # ==========================================

             if score >= self.THRESHOLD:

                 self.log_verification(
                     score,
                     "ACCEPT"
                 )

                 self.result = True

                 self.after(
                     0,
                     self.on_close
                 )

        # ==========================================
        # FACE REJECT
        # ==========================================

             else:

                 self.log_verification(
                     score,
                     "FACE_REJECT"
                 )

                 self._show_error(
                     f"Không nhận diện đúng khuôn mặt.\n\n"
                     f"Face Score: {score:.4f}\n"
                     f"Threshold: {self.THRESHOLD:.4f}"
                 )

         except Exception as e:

             print(
                 "[VERIFICATION ERROR]",
                 e
             )

             self._show_error(
                 f"Lỗi xác thực:\n{e}"
             )

         finally:

             self.is_processing = False

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
    def _update_liveness(self):
         """Chạy vòng kiểm tra Liveness."""

         if not self.is_running:
             return

         if not self.liveness_running:
             return

         try:

        # ==================================================
        # LẤY FRAME
        # ==================================================

             frame = self.camera.get_frame(copy=True)

             if frame is None:

                 print("[LIVENESS] Không lấy được frame")

                 self.after(30,self._update_liveness)

                 return

        # ==================================================
        # FACE DETECTION
        # ==================================================

             bboxes, kpss = self.detector.detect(frame)

             print(
                 "[LIVENESS] Face detected:",
                 0 if bboxes is None else len(bboxes)
             )

             if bboxes is None or len(bboxes) == 0:

                 self.liveness.status = (
                     "Không phát hiện khuôn mặt"
                 )

                 self._show_liveness_frame(frame)

                 self.after(30,self._update_liveness)

                 return

        # ==================================================
        # MULTIPLE FACE
        # ==================================================

             if len(bboxes) > 1:

                 self.liveness.status = (
                     "Chỉ được có 1 khuôn mặt"
                 )

                 self._show_liveness_frame(frame)

                 self.after( 30,self._update_liveness)

                 return

        # ==================================================
        # LANDMARK
        # ==================================================

             if kpss is None or len(kpss) == 0:

                 self.liveness.status = (
                     "Không lấy được landmark"
                 )

                 self._show_liveness_frame(frame)

                 self.after(30,self._update_liveness)

                 return

             landmarks = kpss[0]

        # ==================================================
        # LIVENESS
        # ==================================================

             result, status = self.liveness.update(
                 landmarks
             )

             print(
                 "[DEBUG LIVENESS]",
                 "result =", result,
                 "| step =", self.liveness.current_step,
                 "| sequence =", self.liveness.sequence,
                 "| status =", status)

             self.liveness.status = status

        # ==================================================
        # DISPLAY
        # ==================================================

             self._show_liveness_frame(
                 frame,
                 landmarks
             )

        # ==================================================
        # LIVENESS PASS
        # ==================================================

             if result is True:

                 print(
                     "[LIVENESS] ======================="
                 )

                 print(
                     "[LIVENESS] PASS"
                 )

                 print(
                     "[LIVENESS] Chuyển sang Face Verification"
                 )

                 print(
                     "[LIVENESS] ======================="
                 )

                 self.liveness_running = False
                 self.liveness_passed = True

                 self.btn_verify.configure(
                     text="Đang xác thực khuôn mặt..."
                 )

            # CHỈ TẠI ĐÂY mới được gọi Face Verification
                 threading.Thread(
                     target=self._process_verification,
                     daemon=True
                 ).start()

                 return

        # ==================================================
        # LIVENESS FAIL
        # ==================================================

             if result is False:

                 print(
                     "[LIVENESS] FAIL:",
                     status
                 )

                 self.liveness_running = False
                 self.liveness_passed = False
                 self.is_processing = False

                 self.btn_verify.configure(
                     state="normal",
                     text="Xác nhận khuôn mặt"
                 )

                 messagebox.showerror(
                     "Liveness thất bại",
                     status,
                     parent=self
                 )

                 return

         except Exception as e:

             print(
                 "[LIVENESS ERROR]",
                 repr(e)
             )

             import traceback
             traceback.print_exc()

             self.liveness_running = False
             self.liveness_passed = False
             self.is_processing = False

             self.btn_verify.configure(
                 state="normal",
                 text="Xác nhận khuôn mặt"
             )

             messagebox.showerror(
                 "Lỗi Liveness",
                 str(e),
                 parent=self
             )

             return

    # ==================================================
    # FRAME TIẾP THEO
    # ==================================================

         self.after(30,self._update_liveness)
    def _show_liveness_frame(self,frame,landmarks=None):
         """
         Hiển thị camera + trạng thái liveness.
         """

         display_frame = frame.copy()

    # ==========================================
    # Vẽ landmark
    # ==========================================

         if landmarks is not None:

             for point in landmarks:

                 x, y = point.astype(int)

                 cv2.circle(display_frame,(x, y),3,(0, 255, 0),-1)

    # ==========================================
    # Vẽ trạng thái
    # ==========================================

         display_frame = self.liveness.draw_status(
             display_frame
    )

    # ==========================================
    # Vẽ oval
    # ==========================================

         display_frame = self.draw_oval(
             display_frame
    )

    # ==========================================
    # BGR -> RGB
    # ==========================================

         frame_rgb = cv2.cvtColor(
             display_frame,
             cv2.COLOR_BGR2RGB
         )

         frame_resized = cv2.resize(
             frame_rgb,
             (640, 400),
             interpolation=cv2.INTER_NEAREST
         )

         img = Image.fromarray(
             frame_resized
         )

         self.ctk_img.configure(
             light_image=img,
             dark_image=img
         )

         img.close()