import os
from ultralytics import YOLO
import cv2

# Đường dẫn đến mô hình vừa huấn luyện, tính từ thư mục chứa file này
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "runs", "detect", "vehicle_detection", "weights", "best.pt")

# Kiểm tra nếu model không tồn tại thì báo lỗi rõ ràng
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Không tìm thấy model tại {model_path}. Kiểm tra lại đường dẫn hoặc thư mục runs/detect.")

model = YOLO(model_path)

# Mở webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở webcam. Kiểm tra camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, conf=0.6, iou=0.45, verbose=False)
    annotated_frame = results[0].plot()
    cv2.imshow("Vehicle Detection - Custom Model", annotated_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  
        break

cap.release()
cv2.destroyAllWindows()