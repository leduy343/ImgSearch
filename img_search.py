import cv2
import numpy as np
import pickle
import os
from sklearn.cluster import KMeans
import pandas as pd

# Hàm hiển thị kết quả
def display_results(query_img_path, similar_images, img_dir):
    """
    Hiển thị ảnh truy vấn và ảnh tương tự.
    :param query_img_path: Đường dẫn đến ảnh truy vấn.
    :param similar_images: Danh sách các ảnh tương tự (tên file và khoảng cách).
    :param img_dir: Thư mục chứa ảnh gốc.
    """
    # Hiển thị ảnh truy vấn
    query_img = cv2.imread(query_img_path)
    query_img_resized = cv2.resize(query_img, (600, 600))
    cv2.imshow("Query Image", query_img_resized)

    # Hiển thị các ảnh tương tự
    for img_name, dist in similar_images:
        img_path = os.path.join(img_dir, img_name)
        print(img_path)
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (600, 600))  # Resize ảnh về kích thước 600x600
        cv2.imshow(f"Similar: {img_name} (Distance: {dist:.2f})", img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cluster_features(features_file, num_clusters=3):
    """
    Phân cụm đặc trưng ảnh bằng K-Means.
    
    :param features_file: Đường dẫn đến tệp đặc trưng.
    :param num_clusters: Số cụm cần phân chia.
    :return: KMeans model và từ điển chứa nhãn cụm của từng ảnh.
    """
    with open(features_file, "rb") as f:
        features = pickle.load(f)

    # Lấy tất cả vector đặc trưng
    feature_vectors = []
    img_names = []
    for img_name, data in features.items():
        if data["descriptors"] is not None:
            feature_vectors.append(np.mean(data["descriptors"], axis=0))  # Lấy trung bình các vector
            img_names.append(img_name)

    # Thực hiện phân cụm K-Means
    kmeans = KMeans(n_clusters=num_clusters,n_init=1500, random_state=42)
    cluster_labels = kmeans.fit_predict(feature_vectors)

    # Lưu nhãn cụm tương ứng với từng ảnh
    clustered_data = {img_name: cluster_labels[i] for i, img_name in enumerate(img_names)}
    return kmeans, clustered_data

def find_closest_cluster_and_sort(query_img_path, features_file, kmeans_model, clustered_data, img_dir, algo="SIFT",response_threshold=0.01):
    """
    Tìm cụm gần nhất với ảnh truy vấn và sắp xếp các ảnh trong cụm theo khoảng cách Euclidean.
    :param query_img_path: Đường dẫn đến ảnh truy vấn.
    :param features_file: Đường dẫn đến tệp đặc trưng đã trích xuất.
    :param kmeans_model: Mô hình K-Means đã huấn luyện.
    :param clustered_data: Từ điển nhãn cụm cho từng ảnh.
    :param img_dir: Thư mục chứa ảnh gốc.
    :param algo: Thuật toán sử dụng (SIFT hoặc ORB).
    :return: Danh sách các ảnh trong cụm gần nhất (tên file và khoảng cách).
    """
    # Tải đặc trưng từ tệp
    with open(features_file, "rb") as f:
        features = pickle.load(f)

    # Khởi tạo thuật toán
    if algo == "SIFT":
        detector = cv2.xfeatures2d.SIFT_create()
    elif algo == "ORB":
        detector = cv2.ORB_create()
    else:
        raise ValueError("Algorithm must be 'SIFT' or 'ORB'.")

    # Đọc và xử lý ảnh truy vấn
    query_img = cv2.imread(query_img_path)
    if query_img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {query_img_path}")
    gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    _, query_descriptors = detector.detectAndCompute(gray, None)
    # Tính vector đặc trưng trung bình của ảnh truy vấn
    query_vector = np.mean(query_descriptors, axis=0)

    # Bước 1: Tìm cụm gần nhất
    distances_to_centers = np.linalg.norm(kmeans_model.cluster_centers_ - query_vector, axis=1)
    closest_cluster = np.argmin(distances_to_centers)

    # Bước 2: Lấy ảnh trong cụm gần nhất
    images_in_cluster = [img_name for img_name, cluster in clustered_data.items() if cluster == closest_cluster]

    # Tính khoảng cách giữa ảnh truy vấn và các ảnh trong cụm
    distances_in_cluster = []
    for img_name in images_in_cluster:
        if features[img_name]["descriptors"] is not None:
            feature_vector = np.mean(features[img_name]["descriptors"], axis=0)
            dist = np.linalg.norm(query_vector - feature_vector)
            distances_in_cluster.append((img_name, dist))
    # Bước 3: Sắp xếp các ảnh trong cụm theo khoảng cách tăng dần
    sorted_images = sorted(distances_in_cluster, key=lambda x: x[1])[:10]

    return sorted_images


