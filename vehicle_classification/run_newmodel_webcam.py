"""
Real-Time Vehicle Detection - Sử dụng model từ "new model"
==============================================================
Nhận diện phương tiện giao thông qua Webcam/Camera IP thời gian thực.
Model: D:/html/iot/new model/Real-Time-Vehicle-Detection.../models/best.pt
  - mAP@0.5: 0.975  |  Precision: 91.6%  |  Recall: 93.8%

Sử dụng:
  python run_newmodel_webcam.py               # Webcam mặc định
  python run_newmodel_webcam.py --source 1    # Camera USB thứ 2
  python run_newmodel_webcam.py --source "rtsp://admin:pass@192.168.1.100:554/h264"

Phím tắt:
  q         - Thoát
  s         - Chụp ảnh màn hình và lưu vào thư mục screenshots/
  0-9       - Chuyển camera theo số thứ tự
"""

import cv2
import argparse
import time
import os
from datetime import datetime
from ultralytics import YOLO

# ====================== CẤU HÌNH ======================
MODEL_PATH = "D:/html/iot/new model/Real-Time-Vehicle-Detection-and-Traffic-Flow-Classification-System--main/models/best.pt"
DEFAULT_CONF = 0.4      # Ngưỡng tin cậy tối thiểu (giống mô hình gốc)
DEFAULT_IMGSZ = 640     # Kích thước ảnh đầu vào
HEAVY_TRAFFIC_THRESHOLD = 10   # Số xe để coi là "Kẹt xe"
# =======================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle Detection - New Model (mAP=0.975)")
    parser.add_argument("--source", type=str, default="0",
                        help="Nguồn: 0,1,2 (webcam), rtsp://... (Camera IP), hoặc đường dẫn video/ảnh")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"Ngưỡng confidence (0-1). Mặc định: {DEFAULT_CONF}")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help=f"Kích thước ảnh. Mặc định: {DEFAULT_IMGSZ}")
    return parser.parse_args()


def open_capture(source_str):
    """Mở nguồn video từ webcam index hoặc RTSP URL."""
    try:
        source = int(source_str)
    except ValueError:
        source = source_str
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[LỖI] Không thể mở nguồn: {source_str}")
        return None
    return cap


def draw_overlay(frame, fps, vehicle_counts, total):
    """Vẽ thông tin FPS, số lượng xe và mức độ giao thông lên khung hình."""
    h, w = frame.shape[:2]

    # Nền trong suốt phía trên
    overlay = frame.copy()
    bar_h = 100
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # --- FPS ---
    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 100), 2)

    # --- Tên model ---
    cv2.putText(frame, "Model: New Model (mAP@0.5=0.975)", (12, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # --- Tổng số xe + mức độ giao thông ---
    traffic_status = "KET XE NANG!" if total > HEAVY_TRAFFIC_THRESHOLD else ("DONG DUC" if total > 5 else "THONG THOANG")
    status_color = (0, 0, 255) if total > HEAVY_TRAFFIC_THRESHOLD else ((0, 165, 255) if total > 5 else (0, 220, 0))
    cv2.putText(frame, f"Tong phuong tien: {total}  |  {traffic_status}", (12, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

    # --- Danh sách loại xe bên phải ---
    if vehicle_counts:
        x_right = w - 280
        y_start = 22
        cv2.rectangle(frame, (x_right - 8, 0), (w, bar_h), (0, 0, 0), -1)
        for i, (name, count) in enumerate(vehicle_counts.items()):
            cv2.putText(frame, f"{name}: {count}", (x_right, y_start + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # --- Hướng dẫn phím bên dưới ---
    cv2.putText(frame, "q: Thoat | s: Chup anh | 0-9: Doi camera",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 130), 1)

    return frame


def main():
    args = parse_args()

    # Kiểm tra model
    if not os.path.exists(MODEL_PATH):
        print(f"[LỖI] Không tìm thấy file model tại:\n  {MODEL_PATH}")
        return

    print(f"\n{'='*60}")
    print(f"  NHAN DIEN PHUONG TIEN - NEW MODEL")
    print(f"  mAP@0.5=0.975 | Precision=91.6% | Recall=93.8%")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Source: {args.source}  |  Conf: {args.conf}")
    print(f"{'='*60}")
    print(f"  q=Thoat | s=Chup anh | 0-9=Doi camera")
    print(f"{'='*60}\n")

    # Tải model
    print("[INFO] Đang tải model...")
    model = YOLO(MODEL_PATH)
    class_names = model.names
    print(f"[INFO] Các lớp nhận diện: {list(class_names.values())}")

    # Mở camera
    cap = open_capture(args.source)
    if cap is None:
        return

    # Thư mục lưu ảnh chụp
    screenshot_dir = os.path.join(
        "D:/html/iot/new model/Real-Time-Vehicle-Detection-and-Traffic-Flow-Classification-System--main",
        "screenshots"
    )
    os.makedirs(screenshot_dir, exist_ok=True)

    fps = 0.0
    prev_time = time.time()
    current_source = args.source

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[CẢNH BÁO] Mất kết nối, đang thử lại...")
            cap.release()
            time.sleep(0.5)
            cap = open_capture(current_source)
            if cap is None:
                break
            continue

        # Nhận diện
        results = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)

        # Đếm số xe theo loại
        vehicle_counts = {}
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                name = class_names[cls_id]
                vehicle_counts[name] = vehicle_counts.get(name, 0) + 1
        total_vehicles = sum(vehicle_counts.values())

        # Vẽ bounding box
        annotated = results[0].plot(line_width=2)

        # Tính FPS
        now = time.time()
        fps = 0.85 * fps + 0.15 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        # Vẽ overlay thông tin
        annotated = draw_overlay(annotated, fps, vehicle_counts, total_vehicles)

        # Hiển thị
        cv2.imshow("Vehicle Detection - New Model", annotated)

        # Xử lý phím
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[INFO] Đã thoát.")
            break
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(screenshot_dir, f"detect_{ts}.jpg")
            cv2.imwrite(fname, annotated)
            print(f"[INFO] Đã lưu ảnh: {fname}")
        elif ord('0') <= key <= ord('9'):
            new_src = str(key - ord('0'))
            print(f"[INFO] Chuyển sang camera {new_src}...")
            cap.release()
            cap = open_capture(new_src)
            if cap is None:
                print(f"[LỖI] Không mở được camera {new_src}. Dùng lại camera cũ.")
                cap = open_capture(current_source)
                if cap is None:
                    break
            else:
                current_source = new_src

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã đóng tất cả cửa sổ.")


if __name__ == "__main__":
    main()
