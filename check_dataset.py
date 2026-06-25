from pathlib import Path
from PIL import Image

DATA_DIR = Path("product_dataset")
SPLITS = ["train", "val", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


def main():
    total_all = 0

    for split in SPLITS:
        print(f"\n===== {split.upper()} =====")

        split_dir = DATA_DIR / split
        split_total = 0

        if not split_dir.exists():
            print(f"{split_dir} does not exist.")
            continue

        class_dirs = [p for p in split_dir.iterdir() if p.is_dir()]

        for class_dir in sorted(class_dirs):
            images = [
                p for p in class_dir.iterdir()
                if p.suffix.lower() in IMAGE_EXTS
            ]

            broken = 0

            for img_path in images:
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception:
                    broken += 1
                    print(f"Broken image: {img_path}")

            split_total += len(images)
            print(f"{class_dir.name}: {len(images)} images, broken={broken}")

        total_all += split_total
        print(f"Split total: {split_total}")

    print(f"\nAll images total: {total_all}")


if __name__ == "__main__":
    main()
