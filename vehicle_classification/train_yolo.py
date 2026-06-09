from ultralytics import YOLO

# Đường dẫn đến file data.yaml 
data_yaml = "dataset/data.yaml"

# Tải mô hình YOLOv8 small pretrained (độ chính xác tốt hơn yolov8n)
model = YOLO('yolov8s.pt')

# Huấn luyện
model.train(
    data=data_yaml,
    epochs=100,  # tăng số epoch để model học sâu hơn
    imgsz=640,
    batch=8,
    device='cpu',  # nếu có GPU CUDA, đổi thành '0'
    workers=2,
    name='vehicle_detection'
)

print("Huấn luyện xong! Mô hình ở runs/detect/vehicle_detection/weights/best.pt")