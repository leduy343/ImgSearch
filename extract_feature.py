import cv2
import os
import numpy as np
import pickle

# Hàm trích xuất đặc trưng từ ảnh
def extract_features(img_path, output_path, algo="SIFT", max_keypoints=500,response_threshold=0.01):
    """
    Trích xuất đặc trưng từ ảnh và lưu kết quả vào tệp.
    :param img_path: Đường dẫn thư mục chứa ảnh.
    :param output_path: Đường dẫn lưu trữ tệp đặc trưng.
    :param algo: Thuật toán sử dụng (SIFT hoặc ORB).
    :param max_keypoints: Số lượng điểm đặc trưng tối đa.
    """
    # Lấy danh sách các ảnh
    img_list = [f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    # Khởi tạo thuật toán
    if algo == "SIFT":
        detector = cv2.xfeatures2d.SIFT_create()
    elif algo == "ORB":
        detector = cv2.ORB_create(nfeatures=1500)
    else:
        raise ValueError("Algorithm must be 'SIFT' or 'ORB'.")

    # Lưu trữ đặc trưng
    features = {}

    for img_name in img_list:
        img_file = os.path.join(img_path, img_name)
        print(f"Đang xử lý: {img_file}")

        # Đọc và tiền xử lý ảnh
        img = cv2.imread(img_file)
        if img is None:
            print(f"Không thể đọc ảnh: {img_file}. Bỏ qua...")
            continue

        # Tiền xử lý ảnh
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         # Trích xuất đặc trưng
        keypoints, descriptors = detector.detectAndCompute(gray, None)

        # Lưu đặc trưng
        features[img_name] = {
            "keypoints": [kp.pt for kp in keypoints],  # Chỉ lưu tọa độ điểm đặc trưng
            "descriptors": descriptors
        }

    print(features)

    # Lưu kết quả vào tệp
    with open(output_path, "wb") as f:
        pickle.dump(features, f)

    print(f"Đặc trưng đã được lưu vào: {output_path}")

