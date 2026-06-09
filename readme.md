# Hệ thống Phát hiện và Phân loại Phương tiện Giao thông (Vehicle Detection and Traffic Flow Classification System)


---

##  Các tính năng chính
1. **Phát hiện phương tiện thời gian thực (Webcam/IP Camera)**: Hỗ trợ mở nhiều luồng camera cùng lúc, hiển thị FPS và cảnh báo mức độ kẹt xe dựa trên mật độ giao thông.
2. **Xử lý file video chất lượng cao**: Đọc video từ thư mục test, chạy mô hình phát hiện xe và xuất video đã vẽ khung nhận diện.
3. **Phân loại chi tiết phương tiện (Multi-class)**: Phân biệt cụ thể từng dòng xe bằng mô hình YOLOv8s được huấn luyện sẵn trên tập dữ liệu COCO và mô hình phân loại phụ ResNet-18.
4. **Tối ưu hóa bộ nhớ RAM**: Tích hợp các thuật toán giải phóng bộ nhớ đệm (Garbage Collection, ROI-based overlay) giúp hệ thống xử lý mượt mà các video có độ phân giải siêu cao (2K, 4K, video dọc Portrait) trên CPU/GPU mà không bị lỗi tràn RAM.
5. **Cơ chế Skip thông minh**: Tự động bỏ qua các video đã xử lý thành công để tránh tính toán lại từ đầu khi khởi động lại tiến trình.

---

##  Công nghệ sử dụng
* **Ngôn ngữ lập trình**: Python
* **Học sâu (Deep Learning)**: PyTorch, Torchvision, Ultralytics YOLOv8
* **Thị giác máy tính**: OpenCV (đọc/ghi video, vẽ overlay đồ họa)
* **Tính toán số học**: NumPy (xử lý mảng điểm ảnh)
* **Thuật toán chính**:
  * **YOLOv8** (You Only Look Once v8) cho tác vụ phát hiện vị trí xe (Object Detection).
  * **ResNet-18** (Residual Network 18 layers) cho tác vụ phân loại ảnh xe (Image Classification).

---

##  Cấu trúc thư mục dự án 


```text
iot/
├── vehicle_classification/
│   ├── detect_video_v2.py        # File chính xử lý video (phân loại chi tiết, tối ưu RAM)
│   ├── run_newmodel_webcam.py    # File chính chạy nhận diện qua Webcam thời gian thực
│   ├── detect_webcam_multi.py    # Chạy webcam hỗ trợ đa luồng camera
│   ├── train_classification.py   # Huấn luyện mạng ResNet-18 để phân loại xe
│   ├── train_bdd100k.py          # Huấn luyện mô hình phát hiện đối tượng YOLOv8
│   ├── convert_bdd100k_to_yolo.py# Tiền xử lý dữ liệu nhãn BDD100K sang định dạng YOLO
│   └── detect_video_save.py      # Xử lý video ngầm (bản cũ nhận diện 1 lớp)
├── .gitignore                    # Cấu hình bỏ qua các file rác, file mô hình và zip nặng
└── README.md                     # Tài liệu hướng dẫn sử dụng dự án (File này)
```

---

##  Yêu cầu Hệ thống & Thiết lập Môi trường

Để hệ thống hoạt động ổn định và đạt hiệu năng tốt nhất, bạn cần chuẩn bị môi trường phần cứng và phần mềm như sau:

### 1. Yêu cầu Hệ thống & Phần cứng
* **Hệ điều hành**: Windows 10/11, Ubuntu 20.04+, hoặc macOS.
* **CPU**: Khuyến nghị Intel Core i5 / AMD Ryzen 5 trở lên.
* **RAM**: Tối thiểu 8 GB (Khuyến nghị 16 GB để tránh lỗi tràn bộ nhớ khi chạy video 4K).
* **GPU (Không bắt buộc nhưng khuyến nghị)**: Card đồ họa NVIDIA hỗ trợ CUDA (ví dụ: dòng GTX, RTX) kèm **CUDA Toolkit 11.8 hoặc 12.1** và **cuDNN** tương thích để tăng tốc xử lý thời gian thực.

### 2. Môi trường Python
* **Phiên bản Python**: Khuyến nghị sử dụng **Python 3.8 đến 3.11**

### 3. Các Extension khuyến khích cài đặt trên VS Code
1. **Python** (của Microsoft): Tự động phát hiện trình thông dịch, hỗ trợ debug và chạy file.
2. **Pylance**: Hỗ trợ viết code thông minh, tự động hoàn thành từ (autocomplete) và kiểm tra lỗi cú pháp thực thời.
3. **GitLens**: Quản lý lịch sử commit của Git trực quan và chuyên nghiệp.
4.

---

##  Hướng dẫn Cài đặt & Cấu hình

### Bước 1: Clone dự án về máy tính của bạn
```bash
git clone <URL-KHO-LUA-TRU-CUA-BAN>
cd iot
```

### Bước 2: Tạo và kích hoạt môi trường ảo (Khuyến nghị)
Sử dụng môi trường ảo giúp tránh xung đột thư viện giữa các dự án khác nhau trên máy:
* **Trên Windows (PowerShell / CMD)**:
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
* **Trên macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Bước 3: Cài đặt các thư viện yêu cầu
Sau khi đã kích hoạt môi trường ảo, chạy lệnh sau để cài đặt các package học sâu và thị giác máy tính:
```bash
pip install ultralytics torch torchvision opencv-python numpy
```


---

##  Hướng dẫn Sử dụng

### 1. Nhận diện xe trực tiếp qua Webcam
Mở terminal và chạy lệnh:
```bash
cd vehicle_classification
python run_newmodel_webcam.py --source 0
```
*(Thay số `0` bằng `1`, `2`... nếu bạn sử dụng camera cắm ngoài khác)*.

### 2. Xử lý video mẫu trong thư mục `test_files`

* **Chạy xử lý tất cả video** trong thư mục `test_files`:
  ```bash
  cd vehicle_classification
  python detect_video_v2.py
  ```
  *(Các video kết quả đã vẽ khung phân loại xe sẽ được lưu tự động tại thư mục `test_files/output_v2`)*.

* **Chạy xử lý một video chỉ định** (ví dụ file `14386292_2160_3840_30fps.mp4`):
  ```bash
  cd vehicle_classification
  python detect_video_v2.py --video 14386292_2160_3840_30fps.mp4
  ```
