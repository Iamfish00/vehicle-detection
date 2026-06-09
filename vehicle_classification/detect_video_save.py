"""
Vehicle Detection từ Video - Chế độ lưu file (không cần màn hình)
===================================================================
Xử lý tất cả video trong test_files, lưu kết quả ra file mp4.
Không cần cửa sổ giao diện đồ họa.
"""

import cv2
import os
import time
from ultralytics import YOLO

# ====================== CẤU HÌNH ======================
MODEL_PATH = "D:/html/iot/new model/Real-Time-Vehicle-Detection-and-Traffic-Flow-Classification-System--main/models/best.pt"
VIDEO_DIR  = "D:/html/iot/test_files"
OUTPUT_DIR = "D:/html/iot/test_files/output"
CONF       = 0.4
IMGSZ      = 640
HEAVY_THRESHOLD = 10
# =======================================================


def draw_overlay(frame, total, frame_idx, total_frames, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

    progress = f"Frame: {frame_idx}/{total_frames}"
    cv2.putText(frame, progress, (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    if total > HEAVY_THRESHOLD:
        status, color = "KET XE NANG!", (0, 0, 255)
    elif total > 5:
        status, color = "DONG DUC", (0, 165, 255)
    else:
        status, color = "THONG THOANG", (0, 220, 0)

    cv2.putText(frame, f"Phuong tien: {total}  [{status}]", (12, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)

    # Thanh tiến trình
    if total_frames > 0:
        bar_w = w - 20
        filled = int(bar_w * frame_idx / total_frames)
        cv2.rectangle(frame, (10, h - 8), (10 + bar_w, h - 3), (60, 60, 60), -1)
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

    print(f"  Do phan giai: {orig_w}x{orig_h} | FPS goc: {orig_fps:.1f} | Frames: {total_frames}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_path  = os.path.join(OUTPUT_DIR, f"{base_name}_detected.mp4")
    fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
    writer    = cv2.VideoWriter(out_path, fourcc, orig_fps, (orig_w, orig_h))

    fps = 0.0
    prev_time = time.time()
    frame_idx = 0
    total_vehicles_all = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(frame, conf=CONF, imgsz=IMGSZ, verbose=False)
        total_vehicles = len(results[0].boxes) if results[0].boxes is not None else 0
        total_vehicles_all += total_vehicles

        annotated = results[0].plot(line_width=2)

        now = time.time()
        fps = 0.85 * fps + 0.15 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        annotated = draw_overlay(annotated, total_vehicles, frame_idx, total_frames, fps)
        writer.write(annotated)

        # In tiến độ mỗi 50 frame
        if frame_idx % 50 == 0 or frame_idx == total_frames:
            pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
            avg = total_vehicles_all / frame_idx if frame_idx > 0 else 0
            print(f"  [{pct:5.1f}%] Frame {frame_idx}/{total_frames} | FPS: {fps:.1f} | Xe TB/frame: {avg:.1f}")

    cap.release()
    writer.release()

    avg_vehicles = total_vehicles_all / frame_idx if frame_idx > 0 else 0
    print(f"  [XONG] Da luu: {out_path}")
    print(f"  Tong frames: {frame_idx} | Trung binh xe/frame: {avg_vehicles:.1f}")
    return avg_vehicles


def main():
    print(f"\n{'='*62}")
    print(f"  NHAN DIEN PHUONG TIEN - NEW MODEL (Luu file)")
    print(f"  mAP@0.5=0.975 | Precision=91.6% | Recall=93.8%")
    print(f"{'='*62}")

    print(f"[INFO] Dang tai model...")
    model = YOLO(MODEL_PATH)
    print(f"[INFO] Lop nhan dien: {list(model.names.values())}")
    print(f"[INFO] Output: {OUTPUT_DIR}\n")

    # Lấy tất cả video
    exts = (".mp4", ".avi", ".mov", ".mkv")
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

    start_all = time.time()
    for i, video_path in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] Dang xu ly: {os.path.basename(video_path)}")
        t0 = time.time()
        process_video(model, video_path)
        print(f"  Thoi gian xu ly: {time.time()-t0:.1f}s\n")

    elapsed = time.time() - start_all
    print("="*62)
    print(f"  HOAN THANH! Xu ly {len(video_files)} video trong {elapsed:.0f}s")
    print(f"  Ket qua luu tai: {OUTPUT_DIR}")
    print("="*62)


if __name__ == "__main__":
    main()
