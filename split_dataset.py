from pathlib import Path
import random
import shutil

RAW_DIR = Path("product_dataset/raw")
OUT_DIR = Path("product_dataset")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


def main():
    random.seed(42)

    if not RAW_DIR.exists():
        print(f"Raw dataset folder does not exist: {RAW_DIR}")
        return

    for split in ["train", "val", "test"]:
        split_dir = OUT_DIR / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = [p for p in RAW_DIR.iterdir() if p.is_dir()]
    print(f"Found {len(class_dirs)} classes.\n")

    total_all = 0

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        images = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        ]

        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        split_map = {
            "train": train_imgs,
            "val": val_imgs,
            "test": test_imgs,
        }

        for split, files in split_map.items():
            target_class_dir = OUT_DIR / split / class_name
            target_class_dir.mkdir(parents=True, exist_ok=True)

            for src in files:
                dst = target_class_dir / src.name
                shutil.copy2(src, dst)

        total_all += n

        print(
            f"{class_name}: total={n}, "
            f"train={len(train_imgs)}, "
            f"val={len(val_imgs)}, "
            f"test={len(test_imgs)}"
        )

    print(f"\nTotal images: {total_all}")


if __name__ == "__main__":
    main()
