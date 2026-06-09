"""
Vehicle Detection từ Video - YOLOv8s COCO (Phân loại chi tiết từng loại xe)
=============================================================================
Dùng yolov8s.pt pretrained trên COCO - nhận diện và phân loại chi tiết:
  car, motorcycle, bus, truck, bicycle, train, boat, airplane

Kết quả lưu vào: D:/html/iot/test_files/output_v2/
"""

import cv2
import os
import time
import gc
import torch
from ultralytics import YOLO

# ====================== CẤU HÌNH ĐƯỜNG DẪN TƯƠNG ĐỐI ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Thư mục chứa file script này (vehicle_classification)
ROOT_DIR = os.path.dirname(BASE_DIR)                  # Thư mục gốc dự án (iot)

MODEL_PATH = os.path.join(BASE_DIR, "yolov8s.pt")
VIDEO_DIR  = os.path.join(ROOT_DIR, "test_files")
OUTPUT_DIR = os.path.join(VIDEO_DIR, "output_v2")

CONF       = 0.35     # Ngưỡng confidence
IMGSZ      = 640
HEAVY_THRESHOLD = 10

# Chỉ nhận diện các lớp phương tiện trong COCO
# COCO class IDs: 1=bicycle, 2=car, 3=motorcycle, 5=bus, 6=train, 7=truck, 8=boat, 4=airplane
VEHICLE_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8]

# Màu riêng cho từng loại xe (B, G, R)
CLASS_COLORS = {
    "bicycle":    (255, 165,   0),   # cam
    "car":        (  0, 255,   0),   # xanh lá
    "motorcycle": (  0, 200, 255),   # vàng
    "airplane":   (255,   0, 255),   # tím
    "bus":        (  0,   0, 255),   # đỏ
    "train":      (128,   0, 128),   # tím đậm
    "truck":      (255,  50,  50),   # đỏ nhạt
    "boat":       (200, 100,   0),   # nâu
}

# Tên tiếng Việt
CLASS_VI = {
    "bicycle":    "Xe đạp",
    "car":        "Ô tô",
    "motorcycle": "Xe máy",
    "airplane":   "Máy bay",
    "bus":        "Xe buýt",
    "train":      "Tàu hỏa",
    "truck":      "Xe tải",
    "boat":       "Thuyền",
}
# =======================================================


def draw_overlay(frame, fps, vehicle_counts, total, frame_idx, total_frames):
    h, w = frame.shape[:2]

    # Nền mờ phía trên (chỉ copy và xử lý vùng bar_h để tối ưu hóa bộ nhớ)
    bar_h = min(30 + max(len(vehicle_counts), 1) * 26 + 10, h)
    roi = frame[0:bar_h, 0:w]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.6, roi, 0.4, 0, roi)

    # FPS + Frame
    cv2.putText(frame, f"FPS: {fps:.1f}  |  Frame: {frame_idx}/{total_frames}",
                (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # Từng loại xe
    y = 48
    for name, count in sorted(vehicle_counts.items(), key=lambda x: -x[1]):
        vi_name = CLASS_VI.get(name, name)
        color = CLASS_COLORS.get(name, (255, 255, 255))
        cv2.putText(frame, f"{vi_name}: {count}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        y += 26

    # Tổng + trạng thái giao thông (góc phải)
    if total > HEAVY_THRESHOLD:
        status, s_color = "KET XE NANG!", (0, 0, 255)
    elif total > 5:
        status, s_color = "DONG DUC", (0, 165, 255)
    else:
        status, s_color = "THONG THOANG", (0, 210, 0)

    txt = f"Tong: {total}  [{status}]"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, txt, (w - tw - 12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, s_color, 2)

    # Thanh tiến trình video
    if total_frames > 0:
        bar_w = w - 20
        filled = int(bar_w * frame_idx / total_frames)
        cv2.rectangle(frame, (10, h - 8), (10 + bar_w, h - 3), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, h - 8), (10 + filled, h - 3), (0, 200, 255), -1)

    return frame


def process_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [LOI] Khong mo duoc: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    print(f"  Do phan giai: {orig_w}x{orig_h} | FPS: {orig_fps:.1f} | Frames: {total_frames}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_path  = os.path.join(OUTPUT_DIR, f"{base_name}_v2.mp4")
    fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
    writer    = cv2.VideoWriter(out_path, fourcc, orig_fps, (orig_w, orig_h))

    fps = 0.0
    prev_time = time.time()
    frame_idx = 0
    stats_total = {}   # tổng hợp thống kê

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Nhận diện chỉ các lớp phương tiện (sử dụng torch.no_grad để tránh lưu lịch sử tính toán)
        with torch.no_grad():
            results = model(frame, conf=CONF, imgsz=IMGSZ,
                            classes=VEHICLE_CLASS_IDS, verbose=False)

        # Đếm xe theo loại
        vehicle_counts = {}
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                name = model.names[cls_id]
                vehicle_counts[name] = vehicle_counts.get(name, 0) + 1
                stats_total[name] = stats_total.get(name, 0) + 1

        total_vehicles = sum(vehicle_counts.values())

        # Vẽ bounding box
        annotated = results[0].plot(line_width=2)

        # Tính FPS
        now = time.time()
        fps = 0.85 * fps + 0.15 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        # Vẽ overlay
        annotated = draw_overlay(annotated, fps, vehicle_counts,
                                 total_vehicles, frame_idx, total_frames)
        writer.write(annotated)

        # In tiến độ
        if frame_idx % 50 == 0 or frame_idx == total_frames:
            pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
            summary = ", ".join([f"{CLASS_VI.get(k,k)}: {v}" for k, v in vehicle_counts.items()])
            print(f"  [{pct:5.1f}%] Frame {frame_idx}/{total_frames} | FPS: {fps:.1f} | {summary if summary else 'Khong co xe'}")

        # Giải phóng bộ nhớ và gọi GC định kỳ
        del results
        if frame_idx % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    cap.release()
    writer.release()

    print(f"  [XONG] Luu: {out_path}")
    print(f"  Thong ke toan bo video:")
    for name, count in sorted(stats_total.items(), key=lambda x: -x[1]):
        avg = count / frame_idx if frame_idx > 0 else 0
        print(f"    {CLASS_VI.get(name, name):12s}: tong {count:5d} lan phat hien | TB {avg:.2f}/frame")
    return stats_total


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Vehicle Detection từ Video - YOLOv8s COCO")
    parser.add_argument("--video", type=str, default=None, help="Tên file video cụ thể trong thư mục test_files cần xử lý (ví dụ: 220451_medium.mp4)")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  NHAN DIEN PHUONG TIEN - YOLOv8s COCO (Phan loai chi tiet)")
    print(f"  Cac loai xe: O to, Xe may, Xe buyt, Xe tai, Xe dap, Tau hoa")
    print(f"{'='*65}")

    print(f"[INFO] Dang tai model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"[INFO] Output: {OUTPUT_DIR}\n")

    # Lấy tất cả video hoặc chỉ định video cụ thể
    exts = (".mp4", ".avi", ".mov", ".mkv")
    if args.video:
        target_path = os.path.join(VIDEO_DIR, args.video)
        if os.path.exists(target_path):
            video_files = [target_path]
        else:
            print(f"[LOI] Khong tim thay file video chi dinh: {target_path}")
            return
    else:
        video_files = sorted([
            os.path.join(VIDEO_DIR, f)
            for f in os.listdir(VIDEO_DIR)
            if f.lower().endswith(exts)
        ])

    if not video_files:
        print(f"[LOI] Khong tim thay video trong: {VIDEO_DIR}")
        return

    print(f"[INFO] Tim thay {len(video_files)} video:\n")
    for i, v in enumerate(video_files, 1):
        size_mb = os.path.getsize(v) / 1024 / 1024
        print(f"  {i}. {os.path.basename(v)} ({size_mb:.1f} MB)")
    print()

    all_stats = {}
    start_all = time.time()

    for i, video_path in enumerate(video_files, 1):
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_path  = os.path.join(OUTPUT_DIR, f"{base_name}_v2.mp4")

        # Skip if already processed successfully (size > 1MB)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1024 * 1024:
            print(f"\n[{i}/{len(video_files)}] Da xu ly truoc do: {os.path.basename(video_path)} (Skip)")
            continue

        print(f"\n[{i}/{len(video_files)}] Dang xu ly: {os.path.basename(video_path)}")
        t0 = time.time()
        stats = process_video(model, video_path)
        elapsed = time.time() - t0
        print(f"  Thoi gian: {elapsed:.1f}s")

        # Gộp thống kê toàn bộ
        if stats:
            for k, v in stats.items():
                all_stats[k] = all_stats.get(k, 0) + v

    total_elapsed = time.time() - start_all
    print(f"\n{'='*65}")
    print(f"  HOAN THANH! Xu ly {len(video_files)} video trong {total_elapsed:.0f}s ({total_elapsed/60:.1f} phut)")
    print(f"  Ket qua luu tai: {OUTPUT_DIR}")
    print(f"\n  TONG KET - Tong so lan phat hien tren tat ca video:")
    for name, count in sorted(all_stats.items(), key=lambda x: -x[1]):
        bar = "█" * min(int(count / 100), 40)
        print(f"    {CLASS_VI.get(name, name):12s}: {count:6d}  {bar}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
