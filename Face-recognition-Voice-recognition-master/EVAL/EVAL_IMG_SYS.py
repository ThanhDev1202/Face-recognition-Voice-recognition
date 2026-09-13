import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import re
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from detector.FaceDetector import FaceDetector
from recognizer.FaceRecognizer import FaceRecognizer


def evaluate_lfw_csv(detector, recognizer, dataset_dir, use_detector=True):
    # Tìm file pairs (chấp nhận cả .csv lẫn .txt)
    csv_path = os.path.join(dataset_dir, "pairs.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(dataset_dir, "pairs.txt")

    # Tìm thư mục chứa ảnh
    possible_img_dirs = [
        os.path.join(dataset_dir, "lfw-deepfunneled", "lfw-deepfunneled"),
        os.path.join(dataset_dir, "lfw-deepfunneled"),
        os.path.join(dataset_dir, "lfw")
    ]
    
    images_dir = None
    for d in possible_img_dirs:
        if os.path.exists(d):
            images_dir = d
            break

    if images_dir is None:
        raise FileNotFoundError(f"Không tìm thấy thư mục chứa ảnh LFW trong {dataset_dir}")

    print(f"Đang đọc dữ liệu từ: {csv_path}")
    print(f"Thư mục ảnh: {images_dir}")

    # =========================================================================
    # BỘ LỌC ĐỌC FILE PAIRS CHỐNG LỖI CẤU TRÚC
    # =========================================================================
    pairs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
            
        # Tách chuỗi theo dấu phẩy HOẶC khoảng trắng/tab
        parts = [p.strip() for p in re.split(r'[\t,\s]+', line_clean) if p.strip()]
        
        # Bỏ qua dòng tiêu đề hoặc cấu hình số dòng ở đầu file
        if len(parts) < 3 or parts[0].isdigit() or parts[0].lower() in ['name', 'name1']:
            continue
            
        try:
            if len(parts) == 3:
                # Cặp cùng người: name, img_idx1, img_idx2
                name1 = parts[0]
                name2 = name1
                idx1 = int(parts[1])
                idx2 = int(parts[2])
                label = 1
                pairs.append((name1, idx1, name2, idx2, label))
                
            elif len(parts) >= 4:
                # Cặp khác người: name1, img_idx1, name2, img_idx2
                name1 = parts[0]
                idx1 = int(parts[1])
                name2 = parts[2]
                idx2 = int(parts[3])
                label = 0
                pairs.append((name1, idx1, name2, idx2, label))
        except ValueError:
            # Bỏ qua các dòng lỗi ép kiểu int
            continue

    y_true = []
    y_scores = []
    skipped_count = 0

    total_pairs = len(pairs)
    print(f"Tổng số cặp ảnh hợp lệ đã trích xuất: {total_pairs}\n")

    if total_pairs == 0:
        print("LỖI: Không tìm thấy cặp ảnh nào hợp lệ! Vui lòng kiểm tra lại nội dung file pairs.csv.")
        return

    # Thanh tiến trình %
    pbar = tqdm(pairs, desc="Đang đánh giá LFW", unit="pair")

    for name1, idx1, name2, idx2, label in pbar:
        img1_path = os.path.join(images_dir, name1, f"{name1}_{idx1:04d}.jpg")
        img2_path = os.path.join(images_dir, name2, f"{name2}_{idx2:04d}.jpg")

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            skipped_count += 1
            pbar.set_postfix({"Skipped": skipped_count})
            continue

        try:
            if use_detector:
                _, kpss1 = detector.detect(img1)
                _, kpss2 = detector.detect(img2)

                if len(kpss1) == 0 or len(kpss2) == 0:
                    skipped_count += 1
                    pbar.set_postfix({"Skipped": skipped_count})
                    continue

                emb1 = recognizer.extract_embedding(img1, kpss1[0])
                emb2 = recognizer.extract_embedding(img2, kpss2[0])
            else:
                emb1 = recognizer.extract_embedding(img1)
                emb2 = recognizer.extract_embedding(img2)

            sim = recognizer.compute_cosine_similarity(emb1, emb2)
            y_true.append(label)
            y_scores.append(sim)

        except Exception:
            skipped_count += 1
            pbar.set_postfix({"Skipped": skipped_count})
            continue

    print(f"\n---> Hoàn tất! Bỏ qua {skipped_count} cặp (ảnh lỗi/không thấy mặt).")

    # --- TÍNH TOÁN VÀ IN KẾT QUẢ ---
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    far = fpr
    frr = 1 - tpr
    roc_auc = auc(fpr, tpr)

    eer_idx = np.nanargmin(np.absolute(far - frr))
    best_threshold_eer = thresholds[eer_idx]
    eer_value = (far[eer_idx] + frr[eer_idx]) / 2.0

    best_acc = 0.0
    best_threshold_acc = 0.0
    for t in np.linspace(np.min(y_scores), np.max(y_scores), 1000):
        acc = np.mean((y_scores >= t) == y_true)
        if acc > best_acc:
            best_acc = acc
            best_threshold_acc = t

    print("\n" + "="*45)
    print("        KẾT QUẢ ĐÁNH GIÁ TRÊN DATASET LFW")
    print("="*45)
    print(f"Số cặp đánh giá thành công : {len(y_true)}")
    print(f"Chỉ số AUC                 : {roc_auc:.4f}")
    print(f"Độ chính xác (Accuracy)    : {best_acc * 100:.2f}% (Tại Threshold = {best_threshold_acc:.4f})")
    print(f"Điểm EER (Equal Error Rate): {eer_value * 100:.2f}% (Tại Threshold = {best_threshold_eer:.4f})")
    print(f" -> Mức lỗi tại EER        : FAR = {far[eer_idx]*100:.2f}%, FRR = {frr[eer_idx]*100:.2f}%")
    print("="*45)

    # Biểu đồ
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(thresholds, far, label='FAR (False Accept)', color='red')
    plt.plot(thresholds, frr, label='FRR (False Reject)', color='blue')
    plt.axvline(x=best_threshold_eer, color='green', linestyle='--', label=f'EER Threshold ({best_threshold_eer:.3f})')
    plt.xlabel('Cosine Similarity Threshold')
    plt.ylabel('Rate')
    plt.title('FAR & FRR vs Threshold')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (1 - FRR)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    DATASET_DIR = "DATASET" 

    detector = FaceDetector(model_path="model/det_500m.onnx")
    recognizer = FaceRecognizer(model_path="model/w600k_mbf.onnx")

    evaluate_lfw_csv(detector, recognizer, dataset_dir=DATASET_DIR, use_detector=True)