"""
Vehicle Detection từ Video - New Model (mAP@0.5 = 0.975)
==========================================================
Nhận diện phương tiện giao thông từ video trong thư mục test_files.

Sử dụng:
  python detect_video.py                         # Chạy tất cả video trong test_files
  python detect_video.py --video 220451_medium.mp4  # Chỉ chạy 1 video cụ thể
  python detect_video.py --no-save               # Chỉ xem, không lưu video đầu ra

Kết quả video được lưu vào: D:/html/iot/test_files/output/

Phím tắt khi xem:
  q   - Thoát / sang video tiếp theo
  p   - Tạm dừng / tiếp tục
  s   - Chụp ảnh màn hình
"""

import cv2
import os
import time
import argparse
from datetime import datetime
from ultralytics import YOLO

# ====================== CẤU HÌNH ======================
MODEL_PATH = "D:/html/iot/new model/Real-Time-Vehicle-Detection-and-Traffic-Flow-Classification-System--main/models/best.pt"
VIDEO_DIR  = "D:/html/iot/test_files"
OUTPUT_DIR = "D:/html/iot/test_files/output"
CONF       = 0.4
IMGSZ      = 640
HEAVY_THRESHOLD = 10
# =======================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle Detection từ Video - New Model")
    parser.add_argument("--video", type=str, default=None,
                        help="Tên file video trong test_files (ví dụ: 220451_medium.mp4). "
                             "Nếu không chỉ định sẽ chạy tất cả video.")
    parser.add_argument("--conf", type=float, default=CONF,
                        help=f"Ngưỡng confidence. Mặc định: {CONF}")
    parser.add_argument("--no-save", action="store_true",
                        help="Không lưu video kết quả, chỉ xem trực tiếp.")
    return parser.parse_args()


def draw_overlay(frame, fps, total, frame_idx, total_frames):
    """Vẽ thông tin lên khung hình."""
    h, w = frame.shape[:2]

    # Nền mờ phía trên
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 95), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 100), 2)

    # Tiến trình
    progress = f"Frame: {frame_idx}/{total_frames}" if total_frames > 0 else f"Frame: {frame_idx}"
    cv2.putText(frame, progress, (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Tổng xe + mức độ kẹt xe
    if total > HEAVY_THRESHOLD:
        status, color = "KET XE NANG!", (0, 0, 255)
    elif total > 5:
        status, color = "DONG DUC", (0, 165, 255)
    else:
        status, color = "THONG THOANG", (0, 220, 0)
    cv2.putText(frame, f"Phuong tien: {total}  [{status}]", (12, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)

    # Thanh tiến trình video
    if total_frames > 0:
        bar_w = w - 20
        filled = int(bar_w * frame_idx / total_frames)
        cv2.rectangle(frame, (10, h - 12), (10 + bar_w, h - 5), (60, 60, 60), -1)
        cv2.rectangle(frame, (10, h - 12), (10 + filled, h - 5), (0, 200, 255), -1)

    # Hướng dẫn
    cv2.putText(frame, "q: Tiep theo/Thoat | p: Dung | s: Chup anh",
                (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    return frame


def process_video(model, video_path, save_output=True):
    """Xử lý một file video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [LỖI] Không mở được: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    print(f"  Độ phân giải: {orig_w}x{orig_h}  |  FPS gốc: {orig_fps:.1f}  |  Frames: {total_frames}")

    # Chuẩn bị VideoWriter
    writer = None
    if save_output:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{base_name}_detected.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, orig_fps, (orig_w, orig_h))
        print(f"  Lưu kết quả → {out_path}")

    # Thư mục ảnh chụp màn hình
    screenshot_dir = os.path.join(OUTPUT_DIR, "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)

    fps = 0.0
    prev_time = time.time()
    frame_idx = 0
    paused = False

    window_title = f"Vehicle Detection - {os.path.basename(video_path)}"

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print(f"  [INFO] Hết video.")
                break
            frame_idx += 1

            # Nhận diện
            results = model(frame, conf=CONF, imgsz=IMGSZ, verbose=False)

            # Đếm xe
            total_vehicles = len(results[0].boxes) if results[0].boxes is not None else 0

            # Vẽ bounding box
            annotated = results[0].plot(line_width=2)

            # Tính FPS
            now = time.time()
            fps = 0.85 * fps + 0.15 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            # Vẽ overlay
            annotated = draw_overlay(annotated, fps, total_vehicles, frame_idx, total_frames)

            # Lưu frame
            if writer is not None:
                writer.write(annotated)

        # Hiển thị
        cv2.imshow(window_title, annotated)

        # Xử lý phím
        wait_ms = 0 if paused else 1
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord('q'):
            print(f"  [INFO] Bỏ qua / Thoát.")
            break
        elif key == ord('p'):
            paused = not paused
            print(f"  [INFO] {'Tạm dừng' if paused else 'Tiếp tục'}")
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(screenshot_dir, f"frame_{frame_idx}_{ts}.jpg")
            cv2.imwrite(fname, annotated)
            print(f"  [INFO] Đã lưu ảnh: {fname}")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"  [OK] Đã lưu video kết quả.")
    cv2.destroyWindow(window_title)


def main():
    args = parse_args()
    save_output = not args.no_save

    # Tải model
    print(f"\n{'='*62}")
    print(f"  NHAN DIEN PHUONG TIEN TU VIDEO - NEW MODEL")
    print(f"  mAP@0.5=0.975 | Precision=91.6% | Recall=93.8%")
    print(f"{'='*62}")
    print(f"[INFO] Đang tải model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"[INFO] Lớp nhận diện: {list(model.names.values())}")
    print(f"[INFO] Video output: {'BẬT → ' + OUTPUT_DIR if save_output else 'TẮT'}")
    print(f"{'='*62}\n")

    # Lấy danh sách video cần xử lý
    if args.video:
        video_files = [os.path.join(VIDEO_DIR, args.video)]
        if not os.path.exists(video_files[0]):
            print(f"[LỖI] Không tìm thấy file: {video_files[0]}")
            return
    else:
        exts = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
        video_files = sorted([
            os.path.join(VIDEO_DIR, f)
            for f in os.listdir(VIDEO_DIR)
            if f.lower().endswith(exts)
        ])
        if not video_files:
            print(f"[LỖI] Không tìm thấy video nào trong: {VIDEO_DIR}")
            return

    print(f"[INFO] Tìm thấy {len(video_files)} video:\n")
    for i, v in enumerate(video_files, 1):
        print(f"  {i}. {os.path.basename(v)}")
    print()

    # Xử lý từng video
    for i, video_path in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] Đang xử lý: {os.path.basename(video_path)}")
        process_video(model, video_path, save_output=save_output)
        print()

    print("="*62)
    print("  HOAN THANH! Tất cả video đã được xử lý.")
    if save_output:
        print(f"  Kết quả lưu tại: {OUTPUT_DIR}")
    print("="*62)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
