"""
Vehicle Detection - Multi-Camera Support
=========================================
Nhận diện phương tiện giao thông qua Webcam/Camera IP (RTSP) thời gian thực.

Sử dụng:
  python detect_webcam_multi.py                         # Webcam mặc định (camera 0)
  python detect_webcam_multi.py --source 1              # Camera USB thứ 2
  python detect_webcam_multi.py --source "rtsp://..."   # Camera IP qua RTSP
  python detect_webcam_multi.py --model best.pt         # Dùng mô hình tùy chỉnh

Phím tắt:
  q - Thoát
  s - Chụp ảnh màn hình
  1,2,3... - Chuyển camera (nếu có nhiều camera USB)
"""

from ultralytics import YOLO
import cv2
import argparse
import time
import os
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle Detection - Multi-Camera")
    parser.add_argument(
        "--source", type=str, default="0",
        help="Nguồn video: 0,1,2 (webcam) hoặc rtsp://... (camera IP) hoặc đường dẫn video"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Đường dẫn đến file mô hình (.pt). Mặc định: tự tìm best.pt đã huấn luyện."
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Ngưỡng confidence tối thiểu (0.0 - 1.0). Mặc định: 0.25"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Kích thước ảnh đầu vào. Mặc định: 640"
    )
    return parser.parse_args()


def find_best_model():
    """Tìm file best.pt tốt nhất trong thư mục runs/"""
    # Thử tìm mô hình đã huấn luyện
    candidates = [
        "runs/detect/vehicle_detection/weights/best.pt",
        "runs/detect/train/weights/best.pt",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    # Tìm trong tất cả thư mục runs/detect/
    runs_dir = "runs/detect"
    if os.path.exists(runs_dir):
        for folder in sorted(os.listdir(runs_dir), reverse=True):
            best_path = os.path.join(runs_dir, folder, "weights", "best.pt")
            if os.path.exists(best_path):
                return best_path

    # Dùng mô hình mặc định (YOLOv8 sẽ tự động tải về nếu không có sẵn ở local)
    print("[INFO] Không tìm thấy mô hình huấn luyện riêng. Sử dụng yolov8n.pt (tự động tải từ Ultralytics).")
    return "yolov8n.pt"


def open_source(source_str):
    """Mở nguồn video từ chuỗi source"""
    # Nếu là số nguyên -> webcam index
    try:
        source = int(source_str)
    except ValueError:
        source = source_str  # RTSP URL hoặc đường dẫn file

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[LỖI] Không thể mở nguồn video: {source_str}")
        return None
    return cap


def draw_info_overlay(frame, fps, vehicle_counts, model_name):
    """Vẽ thông tin FPS, số lượng xe lên khung hình"""
    h, w = frame.shape[:2]

    # Vẽ nền bán trong suốt phía trên
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Model name
    cv2.putText(frame, f"Model: {os.path.basename(model_name)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Số lượng phương tiện
    count_text = " | ".join([f"{name}: {count}" for name, count in vehicle_counts.items()])
    if count_text:
        cv2.putText(frame, count_text, (10, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Khong phat hien phuong tien", (10, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 255), 1)

    # Hướng dẫn phím tắt
    cv2.putText(frame, "q: Thoat | s: Chup anh | 0-9: Doi camera",
                (w - 400, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    return frame


def main():
    args = parse_args()

    # Tìm mô hình
    model_path = args.model if args.model else find_best_model()
    if model_path is None:
        print("[LỖI] Không tìm thấy mô hình nào! Hãy huấn luyện trước hoặc chỉ định --model.")
        return

    print(f"[INFO] Đang tải mô hình: {model_path}")
    model = YOLO(model_path)

    # Lấy danh sách tên lớp của mô hình
    class_names = model.names
    print(f"[INFO] Các lớp nhận diện: {class_names}")

    # Mở camera
    print(f"[INFO] Đang kết nối đến nguồn video: {args.source}")
    cap = open_source(args.source)
    if cap is None:
        return

    # Tạo thư mục lưu ảnh chụp
    screenshot_dir = "screenshots"
    os.makedirs(screenshot_dir, exist_ok=True)

    # Biến theo dõi FPS
    prev_time = time.time()
    fps = 0.0

    window_name = f"Vehicle Detection - Source: {args.source}"
    print(f"\n{'='*60}")
    print(f"  NHAN DIEN PHUONG TIEN GIAO THONG - DANG CHAY")
    print(f"  Model: {model_path}")
    print(f"  Source: {args.source}")
    print(f"  Confidence: {args.conf}")
    print(f"{'='*60}")
    print(f"  Nhan phim 'q' de thoat, 's' de chup anh man hinh")
    print(f"{'='*60}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[CẢNH BÁO] Không đọc được frame. Thử kết nối lại...")
            cap.release()
            time.sleep(1)
            cap = open_source(args.source)
            if cap is None:
                break
            continue

        # Chạy nhận diện
        results = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)

        # Đếm số lượng phương tiện theo loại
        vehicle_counts = {}
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = class_names[cls_id]
                vehicle_counts[cls_name] = vehicle_counts.get(cls_name, 0) + 1

        # Vẽ kết quả nhận diện lên ảnh
        annotated_frame = results[0].plot()

        # Tính FPS
        curr_time = time.time()
        fps = 0.8 * fps + 0.2 * (1.0 / max(curr_time - prev_time, 1e-6))
        prev_time = curr_time

        # Vẽ thông tin overlay
        annotated_frame = draw_info_overlay(annotated_frame, fps, vehicle_counts, model_path)

        # Hiển thị
        cv2.imshow(window_name, annotated_frame)

        # Xử lý phím
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[INFO] Thoát chương trình.")
            break
        elif key == ord('s'):
            # Chụp ảnh màn hình
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(screenshot_dir, f"detection_{timestamp}.jpg")
            cv2.imwrite(filename, annotated_frame)
            print(f"[INFO] Đã lưu ảnh: {filename}")
        elif ord('0') <= key <= ord('9'):
            # Chuyển camera
            new_source = str(key - ord('0'))
            print(f"[INFO] Đang chuyển sang camera {new_source}...")
            cap.release()
            cap = open_source(new_source)
            if cap is None:
                print(f"[LỖI] Không thể mở camera {new_source}. Quay lại camera trước.")
                cap = open_source(args.source)
                if cap is None:
                    break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã đóng tất cả cửa sổ.")


if __name__ == "__main__":
    main()
