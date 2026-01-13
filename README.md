# Crowd Counting via Density Map Regression

Final project for Machine Learning - IAI UET 2026

## 1. Giới thiệu

Bài toán **Đếm đám đông (Crowd Counting)** là một bài toán quan trọng trong lĩnh vực Thị giác máy tính (Computer Vision), với mục tiêu ước lượng chính xác số lượng người xuất hiện trong ảnh tĩnh hoặc chuỗi ảnh. Bài toán này có nhiều thách thức do sự phức tạp của môi trường thực tế, bao gồm:

- **Sự thay đổi tỷ lệ (Scale Variation):** Kích thước người thay đổi mạnh do khác biệt về khoảng cách tới camera.
- **Sự che khuất (Occlusion):** Các đối tượng thường che lấp lẫn nhau trong các cảnh đông người.
- **Mật độ phân bố không đồng đều:** Số lượng người có thể chênh lệch lớn giữa các vùng khác nhau trong cùng một ảnh.

Một hướng tiếp cận phổ biến và hiệu quả cho bài toán này là **hồi quy bản đồ mật độ (Density Map Regression)**, trong đó mô hình học cách ánh xạ từ ảnh đầu vào sang một bản đồ mật độ, và tổng tích phân của bản đồ này tương ứng với số lượng người trong ảnh.

---

## 2. Mục tiêu của repository

Repository này được xây dựng nhằm:

- Cài đặt và chạy thử nghiệm **mô hình baseline CSRNet (CVPR 2018)** với trọng số huấn luyện sẵn.
- Minh họa **quy trình suy luận (inference)** cho bài toán đếm đám đông.
- Làm cơ sở thực nghiệm để phục vụ **báo cáo nghiên cứu SOTA**, trong đó kết quả của CSRNet được sử dụng như một mốc so sánh (baseline) với các phương pháp hiện đại hơn (Transformer / Hybrid) dựa trên các kết quả đã được công bố trong tài liệu nghiên cứu.

Lưu ý: Repository **không thực hiện huấn luyện lại mô hình** do giới hạn về tài nguyên tính toán, mà tập trung vào phân tích kiến trúc, quy trình và kết quả suy luận.

---

## 3. Mô hình sử dụng

### CSRNet (CVPR 2018)

- **Loại mô hình:** CNN-based
- **Backbone:** VGG-16
- **Đặc điểm chính:**
CSRNet sử dụng **tích chập giãn nở (Dilated Convolution)** để mở rộng vùng tiếp nhận (receptive field) mà không làm giảm độ phân giải không gian của đặc trưng. Điều này cho phép mô hình khai thác ngữ cảnh rộng trong ảnh, đồng thời vẫn giữ được thông tin chi tiết cần thiết cho việc hồi quy bản đồ mật độ.
- **Vai trò:**
CSRNet là một trong những mô hình baseline mạnh và phổ biến, thường được sử dụng làm mốc chuẩn trong các nghiên cứu Crowd Counting sau này.

---

## 4. Dataset

- **ShanghaiTech Dataset – Part B**
- Đặc trưng:
    - Góc nhìn cao (high-angle surveillance)
    - Mật độ người từ thấp đến trung bình
- Phù hợp với các kịch bản giám sát thực tế và đánh giá khả năng tổng quát hóa của mô hình.

---

## 5. Trọng số huấn luyện (Pre-trained Weights)

Do giới hạn kích thước file của GitHub, file trọng số không được đưa trực tiếp vào repository.

🔗 **Link tải pre-trained weights (Google Drive):**

[https://drive.google.com/file/d/18450x2AHNfZqWKNq1zFiSYJp58HFzpjc/view?usp=sharing](https://drive.google.com/file/d/18450x2AHNfZqWKNq1zFiSYJp58HFzpjc/view?usp=sharing)

Sau khi tải về, đặt file vào thư mục:

```
weights/
└── csrnet_shanghaitech.pth
```

## 6. Cấu trúc thư mục

```python
Crowd-Counting-SOTA-Research/
│

├── samples/                 # Ảnh mẫu để chạy inference

│   ├── test_1.jpg

│   ├── test_2.jpg

│   └── test_3.jpg

│

├── crowd_counting.ipynb     # Notebook chạy inference và trực quan hóa kết quả

├── model_arch.py            # Định nghĩa kiến trúc CSRNet

├── requirements.txt         # Danh sách thư viện cần thiết

├── README.md

├── LICENSE

└── .gitignore
```

## 7. Cài đặt môi trường

Khuyến nghị sử dụng Python ≥ 3.9.

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

Các thư viện chính bao gồm:

- PyTorch
- torchvision
- numpy
- matplotlib
- Pillow
- scipy

## 8. Hướng dẫn chạy

1. Tải file trọng số từ link Google Drive và đặt đúng thư mục `weights/`.
2. Mở file `crowd_counting.ipynb`.
3. Chạy lần lượt các cell để:
    - Nạp mô hình CSRNet
    - Tiền xử lý ảnh
    - Chạy inference
    - Trực quan hóa bản đồ mật độ và số lượng người ước tính

## 9. Ghi chú về nghiên cứu SOTA

Trong báo cáo cuối kỳ, kết quả của CSRNet sẽ được sử dụng làm **baseline** để so sánh với các phương pháp SOTA khác (ví dụ: DM-Count, TransCrowd, CrowdFormer, MAN, …) dựa trên:

- Metric công bố (MAE, MSE)
- Phân tích kiến trúc
- Ưu và nhược điểm của từng phương pháp

Việc so sánh này **dựa trên kết quả từ các bài báo gốc**, không phải từ việc huấn luyện và test trực tiếp trong repository.

---

### Tác giả

**Lê Thị Khánh Linh**

K69 - Ngành Trí tuệ Nhân tạo

Đại học Công nghệ – ĐHQGHN