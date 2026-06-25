from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


DATA_DIR = Path("product_dataset")
MODEL_DIR = Path("logs")
IMAGE_SIZE = 224
BATCH_SIZE = 16


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(num_classes):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def main():
    device = get_device()
    print(f"Using device: {device}")

    with open(MODEL_DIR / "class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)

    num_classes = len(class_names)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    test_dataset = datasets.ImageFolder(
        DATA_DIR / "test",
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = build_model(num_classes)
    model.load_state_dict(
        torch.load(MODEL_DIR / "best_beverage_resnet50.pth", map_location=device)
    )
    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    )

    print("\nClassification Report:")
    print(report)

    cm = confusion_matrix(all_labels, all_preds)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in class_names],
        columns=[f"pred_{c}" for c in class_names]
    )

    MODEL_DIR.mkdir(exist_ok=True)
    cm_df.to_csv(MODEL_DIR / "confusion_matrix.csv", encoding="utf-8-sig")

    correct = sum(int(y_true == y_pred) for y_true, y_pred in zip(all_labels, all_preds))
    total = len(all_labels)
    test_acc = correct / total

    with open(MODEL_DIR / "test_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Test accuracy: {test_acc:.4f}\n\n")
        f.write(report)

    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Saved confusion matrix to {MODEL_DIR / 'confusion_matrix.csv'}")
    print(f"Saved test metrics to {MODEL_DIR / 'test_metrics.txt'}")


if __name__ == "__main__":
    main()
