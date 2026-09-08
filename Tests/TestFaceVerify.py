import os
import sys
import glob
import numpy as np
import cv2

# 1. Thiết lập đường dẫn về thư mục gốc dự án ('python')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

os.chdir(BASE_DIR)

from detector.FaceDetector import FaceDetector
from recognizer.FaceRecognizer import FaceRecognizer


class FaceVerifyTester:

    def __init__(self, det_model_path="model/Face/det_500m.onnx", rec_model_path="model/Face/w600k_mbf.onnx"):
        print("[Init] Đang tải các mô hình AI...")
        self.detector = FaceDetector(model_path=det_model_path, conf_threshold=0.5)
        self.recognizer = FaceRecognizer(model_path=rec_model_path)
        self.THRESHOLD = 0.68  # Ngưỡng Cosine Similarity

    def load_user_embedding_from_db(self, username):
        file_path = os.path.join("database", "face_embeddings", f"{username}.npy")
        if not os.path.exists(file_path):
            print(f"[Error] Không tìm thấy embedding DB cho user '{username}' tại: {file_path}")
            return None

        return np.load(file_path)

    def compute_matrix_similarity(self, live_emb, db_emb):
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

    def verify_image(self, image_path, target_username, threshold=None):
        if threshold is None:
            threshold = self.THRESHOLD

        # Đọc file bằng np.fromfile + cv2.imdecode để tránh lỗi Unicode tiếng Việt
        try:
            img_array = np.fromfile(image_path, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception:
            frame = None

        if frame is None:
            print(f"[Error] Không thể đọc file: {image_path}")
            return "ERROR", 0.0

        bboxes, kpss = self.detector.detect(frame)
        if len(bboxes) == 0:
            print(f"[{os.path.basename(image_path)}] -> NO FACE ⚠️ (Không thấy mặt)")
            return "NO_FACE", 0.0

        # Lấy khuôn mặt có điểm tin cậy cao nhất
        bbox = bboxes[0]
        landmarks = kpss[0]

        embedding_live = self.recognizer.extract_embedding(frame, landmarks)
        norm_live = np.linalg.norm(embedding_live)
        if norm_live > 0:
            embedding_live = embedding_live / norm_live

        embedding_db = self.load_user_embedding_from_db(target_username)
        if embedding_db is None:
            return "ERROR", 0.0

        score = self.compute_matrix_similarity(embedding_live, embedding_db)
        is_match = score >= threshold
        status_text = "ACCEPT ✅" if is_match else "REJECT ❌"
        result_code = "ACCEPT" if is_match else "REJECT"

        print(f"[{os.path.basename(image_path)}] -> Score: {score:.4f} | Result: {status_text}")
        return result_code, score


if __name__ == "__main__":
    tester = FaceVerifyTester()

    TARGET_USER = "Thanh"  
    DATASET_DIR = "C:/Users/admin/OneDrive/Desktop/Dataset/Face"

    # Lọc các tập tin ảnh và loại bỏ tuyệt đối các đường dẫn trùng lặp bằng set()
    extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths_set = set()

    for ext in extensions:
        found_files = glob.glob(os.path.join(DATASET_DIR, ext))
        for file_path in found_files:
            image_paths_set.add(os.path.normpath(file_path))

    image_paths = sorted(list(image_paths_set))

    print("\n" + "=" * 60)
    print(f"BẮT ĐẦU KÍCH HOẠT TEST BỘ DỮ LIỆU CỦA USER: '{TARGET_USER}'")
    print(f"Tổng số ảnh thực tế tìm thấy: {len(image_paths)}")
    print("=" * 60)

    # Khởi tạo các biến đếm thống kê
    accept_count = 0
    reject_count = 0
    no_face_count = 0
    error_count = 0

    for img_path in image_paths:
        result_code, score = tester.verify_image(img_path, target_username=TARGET_USER)
        
        if result_code == "ACCEPT":
            accept_count += 1
        elif result_code == "REJECT":
            reject_count += 1
        elif result_code == "NO_FACE":
            no_face_count += 1
        else:
            error_count += 1

    total_images = len(image_paths)

    if total_images > 0:
        # Tỷ lệ chấp nhận đúng trên tổng số ảnh trong tập dữ liệu
        overall_accuracy = (accept_count / total_images) * 100
        
        # Tỷ lệ chấp nhận chỉ tính trên các ảnh có phát hiện mặt
        detected_images = accept_count + reject_count
        detection_accuracy = (accept_count / detected_images * 100) if detected_images > 0 else 0.0

        print("\n" + "=" * 60)
        print(f"KẾT QUẢ ĐÁNH GIÁ THỐNG KÊ CHO USER '{TARGET_USER}':")
        print(f"- Tổng số ảnh kiểm tra: {total_images}")
        print(f"- Nhận diện đúng (ACCEPT): {accept_count}")
        print(f"- Sai khác/Bị từ chối (REJECT): {reject_count}")
        print(f"- Không phát hiện được mặt (NO FACE): {no_face_count}")
        if error_count > 0:
            print(f"- Lỗi đọc file/DB (ERROR): {error_count}")
        print("-" * 60)
        print(f"- Độ chính xác toàn bộ tập dữ liệu (Overall Accuracy): {overall_accuracy:.2f}%")
        print(f"- Độ chính xác trên các ảnh detect được mặt: {detection_accuracy:.2f}%")
        print("=" * 60)