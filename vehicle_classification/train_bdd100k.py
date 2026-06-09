import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='Huấn luyện YOLOv8 trên BDD100K với checkpoint/resume')
    parser.add_argument('--model', default='yolov8s.pt', help='Mô hình YOLO ban đầu')
    parser.add_argument('--data', default='dataset/bdd100k_data.yaml', help='Đường dẫn file data YAML')
    parser.add_argument('--epochs', type=int, default=20, help='Số epoch cho lần chạy này')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Kích thước ảnh')
    parser.add_argument('--device', default='cpu', help="'cpu' hoặc '0' nếu dùng GPU")
    parser.add_argument('--fraction', type=float, default=1.0, help='Tỉ lệ dữ liệu để train (0.1=10%%)')
    parser.add_argument('--name', default='bdd100k_train', help='Tên run lưu trong runs/detect')
    parser.add_argument('--save-period', type=int, default=1, help='Lưu checkpoint mỗi n epoch')
    parser.add_argument('--notify-images', type=int, default=20, help='Thông báo mỗi khi xử lý đủ số ảnh này')
    parser.add_argument('--resume', action='store_true', help='Resume nếu đã có checkpoint last.pt')
    return parser.parse_args()


class ImageProgressCallback:
    def __init__(self, notify_every: int):
        self.notify_every = notify_every
        self.processed_images = 0
        self.next_notify = notify_every

    def __call__(self, trainer):
        batch_size = getattr(trainer, 'batch_size', None) or getattr(trainer.args, 'batch', 0)
        self.processed_images += batch_size
        if self.processed_images >= self.next_notify:
            print(f"[Checkpoint] Đã xử lý khoảng {self.processed_images} ảnh.")
            self.next_notify += self.notify_every


def main():
    args = parse_args()
    weights_dir = Path('runs') / 'detect' / args.name / 'weights'
    last_checkpoint = weights_dir / 'last.pt'
    resume_flag = args.resume and last_checkpoint.exists()

    print('----- Training BDD100K -----')
    print(f'model: {args.model}')
    print(f'data: {args.data}')
    print(f'epochs: {args.epochs}')
    print(f'batch: {args.batch}')
    print(f'imgsz: {args.imgsz}')
    print(f'device: {args.device}')
    print(f'fraction: {args.fraction}')
    print(f'name: {args.name}')
    print(f'save_period: {args.save_period}')
    print(f'resume: {resume_flag}')

    if args.resume and not resume_flag:
        print('Warning: Không tìm thấy checkpoint last.pt, training sẽ bắt đầu lại từ pretrained.')

    model = YOLO(args.model)
    progress_callback = ImageProgressCallback(args.notify_images)
    model.add_callback('on_train_batch_end', progress_callback)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        fraction=args.fraction,
        name=args.name,
        save_period=args.save_period,
        save=True,
        resume=resume_flag,
        cache=True,
        workers=2,
        exist_ok=True,
    )

    print('Huấn luyện xong! Kiểm tra thư mục runs/detect/')


if __name__ == '__main__':
    main()
