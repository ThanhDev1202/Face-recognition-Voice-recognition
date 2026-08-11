import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
# ============================================================
# 1. XÁC ĐỊNH THƯ MỤC GỐC PROJECT
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
sys.path.append(BASE_DIR)


# ============================================================
# 2. IMPORT MODEL CỦA APP
# ============================================================

from detector.FaceDetector import FaceDetector
from recognizer.FaceRecognizer import FaceRecognizer


# ============================================================
# 3. CẤU HÌNH
# ============================================================

# User đã đăng ký trong database
USERNAME = "thanh"

# Threshold hiện tại của app
THRESHOLD = 0.80


# ============================================================
# 4. ĐƯỜNG DẪN DATASET LFW
# ============================================================

LFW_DIR = os.path.join(
    BASE_DIR,
    "DATASET",
    "lfw-deepfunneled",
    "lfw-deepfunneled"
)


# ============================================================
# 5. ĐƯỜNG DẪN EMBEDDING DATABASE
# ============================================================

EMBEDDING_PATH = os.path.join(
    BASE_DIR,
    "database",
    "face_embeddings",
    f"{USERNAME}.npy"
)


# ============================================================
# 6. FILE KẾT QUẢ
# ============================================================

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "far_results.csv"
)


# ============================================================
# 7. KIỂM TRA ĐƯỜNG DẪN
# ============================================================

print("=" * 60)
print("KIỂM TRA ĐƯỜNG DẪN")
print("=" * 60)

print("BASE_DIR:")
print(BASE_DIR)

print("\nLFW_DIR:")
print(LFW_DIR)

print("\nEMBEDDING:")
print(EMBEDDING_PATH)


if not os.path.exists(LFW_DIR):

    raise FileNotFoundError(
        "\nKhông tìm thấy thư mục LFW!\n"
        f"Đường dẫn đang kiểm tra:\n{LFW_DIR}\n\n"
        "Hãy kiểm tra cấu trúc:\n"
        "DATASET/\n"
        "└── lfw-deepfunneled/"
    )


if not os.path.exists(EMBEDDING_PATH):

    raise FileNotFoundError(
        "\nKhông tìm thấy embedding của user!\n"
        f"Đường dẫn:\n{EMBEDDING_PATH}"
    )


print("\n✓ Tìm thấy dataset")
print("✓ Tìm thấy embedding")


# ============================================================
# 8. LOAD EMBEDDING ĐÃ ĐĂNG KÝ
# ============================================================

embedding_db = np.load(
    EMBEDDING_PATH
).astype(np.float32)


print("\nShape embedding DB:", embedding_db.shape)


# Chuẩn hóa embedding
norm = np.linalg.norm(embedding_db)

if norm == 0:

    raise ValueError(
        "Embedding trong database có norm = 0!"
    )

embedding_db = embedding_db / norm


# ============================================================
# 9. LOAD FACE DETECTOR
# ============================================================

print("\n" + "=" * 60)
print("ĐANG LOAD FACE DETECTOR")
print("=" * 60)

detector = FaceDetector(
    model_path=os.path.join(
        BASE_DIR,
        "model",
        "det_500m.onnx"
    ),
    conf_threshold=0.5
)


# ============================================================
# 10. LOAD FACE RECOGNIZER
# ============================================================

print("\n" + "=" * 60)
print("ĐANG LOAD MOBILEFACENET")
print("=" * 60)

recognizer = FaceRecognizer(
    model_path=os.path.join(
        BASE_DIR,
        "model",
        "w600k_mbf.onnx"
    )
)


# ============================================================
# 11. LẤY DANH SÁCH ẢNH LFW
# ============================================================

image_paths = []


for person_name in os.listdir(LFW_DIR):

    person_dir = os.path.join(
        LFW_DIR,
        person_name
    )

    # Bỏ qua file
    if not os.path.isdir(person_dir):
        continue

    for filename in os.listdir(person_dir):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            image_path = os.path.join(
                person_dir,
                filename
            )

            image_paths.append(
                {
                    "person": person_name,
                    "path": image_path
                }
            )


print("\n" + "=" * 60)
print("THÔNG TIN DATASET")
print("=" * 60)

print(
    f"Tổng số ảnh tìm thấy: {len(image_paths)}"
)

print(
    f"User trong database: {USERNAME}"
)

print(
    f"Threshold: {THRESHOLD}"
)


# ============================================================
# 12. BIẾN THỐNG KÊ
# ============================================================

total_attempts = 0

false_accepts = 0

true_rejects = 0

no_face_count = 0

multiple_face_count = 0

error_count = 0


results = []


# ============================================================
# 13. CHẠY TOÀN BỘ ẢNH LFW
# ============================================================

print("\n" + "=" * 60)
print("BẮT ĐẦU ĐÁNH GIÁ FAR")
print("=" * 60)


for item in tqdm(
    image_paths,
    desc="Processing LFW"
):

    person_name = item["person"]
    image_path = item["path"]

    # --------------------------------------------------------
    # Đọc ảnh
    # --------------------------------------------------------

    frame = cv2.imread(image_path)

    if frame is None:

        error_count += 1

        results.append({
            "person": person_name,
            "image": image_path,
            "score": "",
            "threshold": THRESHOLD,
            "result": "ERROR_READ_IMAGE"
        })

        continue


    try:

        # ====================================================
        # 14. FACE DETECTION
        # Giống FaceVerifyWindow
        # ====================================================

        bboxes, kpss = detector.detect(frame)


        # ----------------------------------------------------
        # Không phát hiện khuôn mặt
        # ----------------------------------------------------

        if len(bboxes) == 0:

            no_face_count += 1

            results.append({
                "person": person_name,
                "image": image_path,
                "score": "",
                "threshold": THRESHOLD,
                "result": "NO_FACE"
            })

            continue


        # ----------------------------------------------------
        # Có nhiều khuôn mặt
        # ----------------------------------------------------

        if len(bboxes) > 1:

            multiple_face_count += 1

            results.append({
                "person": person_name,
                "image": image_path,
                "score": "",
                "threshold": THRESHOLD,
                "result": "MULTIPLE_FACES"
            })

            continue


        # ====================================================
        # 15. LẤY LANDMARKS
        # Giống FaceVerifyWindow
        # ====================================================

        landmarks = kpss[0]


        # ====================================================
        # 16. EXTRACT EMBEDDING
        # Giống FaceVerifyWindow
        # ====================================================

        embedding_live = recognizer.extract_embedding(
            frame,
            landmarks
        )


        # ====================================================
        # 17. COSINE SIMILARITY
        # Giống FaceVerifyWindow
        # ====================================================

        score = recognizer.compute_cosine_similarity(
            embedding_live,
            embedding_db
        )


        # ====================================================
        # 18. SO SÁNH THRESHOLD
        # ====================================================

        if score >= THRESHOLD:

            result = "ACCEPT"

            # Người khác nhưng ACCEPT
            # => FALSE ACCEPTANCE

            false_accepts += 1

        else:

            result = "REJECT"

            # Người khác và REJECT
            # => TRUE REJECT

            true_rejects += 1


        total_attempts += 1


        # ====================================================
        # 19. LƯU KẾT QUẢ
        # ====================================================

        results.append({
            "person": person_name,
            "image": image_path,
            "score": float(score),
            "threshold": THRESHOLD,
            "result": result
        })


    except Exception as e:

        error_count += 1

        results.append({
            "person": person_name,
            "image": image_path,
            "score": "",
            "threshold": THRESHOLD,
            "result": f"ERROR: {str(e)}"
        })


# ============================================================
# 20. TÍNH FAR
# ============================================================

if total_attempts > 0:

    FAR = false_accepts / total_attempts

else:

    FAR = 0.0


# ============================================================
# 21. LƯU CSV
# ============================================================

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 22. IN KẾT QUẢ
# ============================================================

print("\n")

print("=" * 60)
print("          KẾT QUẢ ĐÁNH GIÁ FAR")
print("=" * 60)

print(
    f"User đăng ký             : {USERNAME}"
)

print(
    f"Threshold                : {THRESHOLD:.4f}"
)

print(
    f"Tổng ảnh LFW             : {len(image_paths)}"
)

print(
    f"Lần thử hợp lệ           : {total_attempts}"
)

print(
    f"False Acceptance         : {false_accepts}"
)

print(
    f"True Reject              : {true_rejects}"
)

print(
    f"Không phát hiện mặt      : {no_face_count}"
)

print(
    f"Nhiều khuôn mặt          : {multiple_face_count}"
)

print(
    f"Lỗi                      : {error_count}"
)

print("-" * 60)

print(
    f"FAR                      : {FAR:.6f}"
)

print(
    f"FAR (%)                  : {FAR * 100:.4f}%"
)

print("=" * 60)

print(
    "\nFile kết quả:"
)

print(
    OUTPUT_FILE
)