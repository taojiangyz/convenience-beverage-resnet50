from pathlib import Path
import json
import sys

from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models


MODEL_DIR = Path("logs")
IMAGE_SIZE = 224


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


def load_model(device):
    class_names_path = MODEL_DIR / "class_names.json"
    model_path = MODEL_DIR / "best_beverage_resnet50.pth"

    if not class_names_path.exists():
        raise FileNotFoundError(f"Class names file not found: {class_names_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    model = build_model(len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model, class_names


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    return tensor


def predict(image_path):
    device = get_device()
    print(f"Using device: {device}")

    model, class_names = load_model(device)

    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_k = min(3, len(class_names))
    top_probs, top_indices = torch.topk(probabilities, k=top_k)

    pred_idx = top_indices[0].item()
    pred_class = class_names[pred_idx]
    confidence = top_probs[0].item()

    print(f"\nImage: {image_path}")
    print(f"Prediction: {pred_class}")
    print(f"Confidence: {confidence:.4f}")

    print("\nTop-3 predictions:")
    for prob, idx in zip(top_probs, top_indices):
        print(f"{class_names[idx.item()]}: {prob.item():.4f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 predict_one.py path/to/image.jpg")
        return

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Image file not found: {image_path}")
        return

    predict(image_path)


if __name__ == "__main__":
    main()
