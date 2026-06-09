import json
from pathlib import Path
from PIL import Image

# Thay đổi đường dẫn nếu bạn giải nén BDD100K ở vị trí khác
BDD_ROOT = Path(r"D:\html\iot\archive (1)\bdd100k\bdd100k")
LABELS_META = Path(r"D:\html\iot\archive (1)\bdd100k_labels_release\bdd100k\labels")
OUTPUT_ROOT = BDD_ROOT / "labels" / "10k"

CATEGORY_MAP = {
    "bus": 0,
    "car": 1,
    "motorcycle": 2,
    "truck": 3,
}

SPLITS = [
    (
        "train",
        LABELS_META / "bdd100k_labels_images_train.json",
        BDD_ROOT / "images" / "10k" / "train",
    ),
    (
        "val",
        LABELS_META / "bdd100k_labels_images_val.json",
        BDD_ROOT / "images" / "10k" / "val",
    ),
]

NAMES = ["Bus", "Car", "Motorcycle", "Truck"]


def convert_split(split_name, json_path, images_root):
    if not json_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file annotation: {json_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {images_root}")

    output_dir = OUTPUT_ROOT / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    sizes = {}
    total = 0
    skipped = 0

    for item in data:
        img_name = item["name"]
        img_path = images_root / img_name
        if not img_path.exists():
            # Nếu ảnh không ở trong split đúng, tìm trong các thư mục con cùng cấp.
            parent_dir = images_root.parent
            if parent_dir.exists():
                for sibling in parent_dir.iterdir():
                    candidate = sibling / img_name
                    if candidate.exists():
                        img_path = candidate
                        break

        if not img_path.exists():
            skipped += 1
            continue

        if img_path not in sizes:
            with Image.open(img_path) as img:
                sizes[img_path] = img.size

        width, height = sizes[img_path]
        out_path = output_dir / f"{img_path.stem}.txt"
        lines = []

        for label in item.get("labels", []):
            category = label.get("category")
            if category not in CATEGORY_MAP:
                continue

            box = label.get("box2d")
            if not box:
                continue

            x1 = float(box.get("x1", 0.0))
            y1 = float(box.get("y1", 0.0))
            x2 = float(box.get("x2", 0.0))
            y2 = float(box.get("y2", 0.0))
            if x2 <= x1 or y2 <= y1:
                continue

            cx = (x1 + x2) / 2.0 / width
            cy = (y1 + y2) / 2.0 / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height

            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= bw <= 1 and 0 <= bh <= 1):
                continue

            class_id = CATEGORY_MAP[category]
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        total += 1

    print(f"{split_name}: đã tạo {total} label files, bỏ qua {skipped} ảnh không tìm thấy.")


if __name__ == "__main__":
    print("Chuyển đổi BDD100K sang định dạng YOLO...")
    for split_name, json_file, images_root in SPLITS:
        convert_split(split_name, json_file, images_root)
    print("Hoàn thành. Labels được lưu tại:")
    print(OUTPUT_ROOT)
    print("Sau đó dùng dataset/bdd100k_data.yaml để huấn luyện.")
