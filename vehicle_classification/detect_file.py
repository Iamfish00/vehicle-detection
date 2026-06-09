import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='Chạy kiểm tra ảnh/video với model YOLO đã huấn luyện')
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Đường dẫn tới file ảnh hoặc video cần kiểm tra'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Đường dẫn tới file .pt của model. Mặc định tìm best.pt/last.pt trong runs/detect/bdd100k_train/weights'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Kích thước ảnh vào khi infer'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
            help='Ngưỡng confidence để hiển thị kết quả (mặc định 0.5 để lọc detection yếu)'
    )
    return parser.parse_args()


def find_default_model():
    candidates = [
        Path('../new model/Real-Time-Vehicle-Detection-and-Traffic-Flow-Classification-System--main/models/best.pt'),
        Path('runs/detect/bdd100k_train/weights/best.pt'),
        Path('runs/detect/bdd100k_train/weights/last.pt'),
        Path('runs/detect/bdd100k_train/weights/epoch0.pt'),
    ]
    for idx, candidate in enumerate(candidates, 1):
        if candidate.exists():
            print(f'>> Tìm thấy model: {candidate.resolve()}')
            print(f'   ({idx}. {candidate})')
            return candidate
    raise FileNotFoundError(
        'Không tìm thấy mô hình nào! Kiểm tra đường dẫn hoặc chỉ định --model'
    )


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def is_video(path: Path) -> bool:
    return path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.webm'}


def show_image(image, window_name='Result'):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_detections_with_class_labels(frame, results, model_names):
    """Vẽ bounding box với tên lớp và confidence score rõ ràng hơn"""
    annotated = frame.copy()
    
    if results[0].boxes is not None:
        boxes = results[0].boxes
        print(f'\n>> Phát hiện {len(boxes)} đối tượng:')
        
        for idx, box in enumerate(boxes):
            # Lấy tọa độ
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            
            # Lấy class ID và confidence
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model_names[cls_id] if cls_id < len(model_names) else f'Class {cls_id}'
            
            # In thông tin
            print(f'   {idx+1}. {cls_name} (Confidence: {conf:.2%})')
            
            # Vẽ bounding box
            color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)  # Green nếu cao, Orange nếu thấp
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ nhãn
            label = f'{cls_name}: {conf:.0%}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0] + 5, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return annotated

def main():
    args = parse_args()
    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f'Không tìm thấy file source: {source_path}')

    model_path = Path(args.model) if args.model else find_default_model()
    if not model_path.exists():
        raise FileNotFoundError(f'Không tìm thấy model: {model_path}')

    print(f'>> Source: {source_path}')
    print(f'>> Model: {model_path}')
    print(f'>> Image size: {args.imgsz}, Confidence: {args.conf}')

    model = YOLO(str(model_path))
    model_names = model.names
    
    output_dir = Path('runs/detect/test_results')
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_image(source_path):
        frame = cv2.imread(str(source_path))
        if frame is None:
            raise ValueError(f'Không thể đọc ảnh: {source_path}')

        results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        annotated = draw_detections_with_class_labels(frame, results, model_names)
        save_path = output_dir / f'{source_path.stem}_annotated{source_path.suffix}'
        cv2.imwrite(str(save_path), annotated)
        print('>> Đã lưu ảnh kết quả:', save_path)
        show_image(annotated, window_name='YOLO Inference Image')

    elif is_video(source_path):
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise ValueError(f'Không thể mở video: {source_path}')

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        save_path = output_dir / f'{source_path.stem}_annotated.mp4'
        writer = cv2.VideoWriter(
            str(save_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        window_name = 'YOLO Inference Video'
        print('>> Nhấn q để dừng trước khi kết thúc video.')

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            annotated = draw_detections_with_class_labels(frame, results, model_names)
            writer.write(annotated)
            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('>> Dừng video theo lệnh người dùng.')
                break

        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print('>> Đã lưu video kết quả:', save_path)

    else:
        raise ValueError('File không phải ảnh hoặc video. Vui lòng dùng định dạng ảnh/video hợp lệ.')


if __name__ == '__main__':
    main()
