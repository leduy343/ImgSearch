import streamlit as st
import os
from PIL import Image
from img_search import cluster_features, find_closest_cluster_and_sort
from extract_feature import extract_features
import math
def display_results(query_img_path, similar_images, img_dir):
    st.image(query_img_path, caption="Ảnh truy vấn", use_container_width=True)
    st.write("### Kết quả tìm kiếm:")

    # Kích thước cố định cho tất cả ảnh hiển thị
    fixed_size = (500, 400)

    # Số ảnh tối đa mỗi hàng
    images_per_row = 5

    # Chia danh sách các ảnh thành các nhóm với mỗi nhóm có tối đa 5 ảnh
    num_rows = math.ceil(len(similar_images) / images_per_row)
    
    # Hiển thị từng hàng
    for row in range(num_rows):
        start_index = row * images_per_row
        end_index = min((row + 1) * images_per_row, len(similar_images))
        
        # Tạo các cột cho một hàng
        cols = st.columns(end_index - start_index)
        
        # Hiển thị ảnh trong các cột tương ứng
        for i, (img_name, dist) in enumerate(similar_images[start_index:end_index]):
            img_path = os.path.join(img_dir, img_name)
            
            # Mở và thay đổi kích thước ảnh
            img = Image.open(img_path).resize(fixed_size)
            
            # Hiển thị ảnh trong cột tương ứng
            cols[i].image(img, caption=f"{img_name}\nDistance: {dist:.2f}", use_container_width=True)



# Giao diện Streamlit
st.title("Tìm kiếm ảnh tương tự")
st.write("Ứng dụng tìm kiếm ảnh tương tự dựa trên đặc trưng SIFT hoặc ORB.")

# Chọn thư mục ảnh
img_dir = st.text_input("Thư mục chứa ảnh", "test_img")
algo = st.selectbox("Chọn thuật toán trích xuất đặc trưng", ["ORB", "SIFT"])

# Đặt tên tệp CSV theo thuật toán
csv_file = f"features_{algo.lower()}.csv"

# Lưu đặc trưng
if st.button(f"Lưu đặc trưng {algo} vào CSV"):
    if os.path.exists(img_dir):
        try:
            extract_features(img_dir, csv_file, algo, max_keypoints=500)
            # Hiển thị thông báo thành công
            st.success(f"Đặc trưng đã được lưu thành công vào tệp: {csv_file}")
        except Exception as e:
            st.error(f"Lỗi trong quá trình lưu đặc trưng: {e}")
    else:
        st.error("Thư mục ảnh không tồn tại!")

# Upload ảnh truy vấn
uploaded_file = st.file_uploader("Chọn ảnh truy vấn", type=["jpg", "jpeg", "png"])
find_button = st.button("Tìm kiếm ảnh tương tự")

if uploaded_file and find_button:
    # Lưu ảnh truy vấn tạm thời
    query_img_path = f"temp_{uploaded_file.name}"
    with open(query_img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        
        kmeans, clustered_data = cluster_features(csv_file , num_clusters=10)
        img_dir = "test_img/"
        sorted_images = find_closest_cluster_and_sort(
            query_img_path=query_img_path,
            features_file=csv_file ,
            kmeans_model=kmeans,
            clustered_data=clustered_data,
            img_dir=img_dir,
            algo= algo
        )

        display_results(query_img_path, sorted_images, img_dir)

    except Exception as e:
        st.error(f"Lỗi: {str(e)}")
