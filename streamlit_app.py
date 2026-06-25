from pathlib import Path
import json

from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import streamlit as st


MODEL_DIR = Path("logs")
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.60


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


@st.cache_resource
def load_model():
    device = get_device()

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

    return model, class_names, device


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor


def predict(image):
    model, class_names, device = load_model()

    image_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_k = min(3, len(class_names))
    top_probs, top_indices = torch.topk(probabilities, k=top_k)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "class_name": class_names[idx.item()],
            "probability": prob.item()
        })

    return results, device


def main():
    st.set_page_config(
        page_title="Beverage Image Classification",
        page_icon="🍵",
        layout="centered"
    )

    st.markdown(
        """
        <div style="line-height: 1.3; margin-bottom: 0.5rem;">
            <div style="font-size: 28px; font-weight: 700;">🍵 Convenience Store Beverage Classifier</div>
            <div style="font-size: 24px; font-weight: 700;">コンビニ飲料画像分類デモ</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption("ResNet50-based image classification demo / ResNet50を用いた画像分類デモ")

    st.write(
        "Upload a convenience store beverage image, and the model will predict "
        "which trained beverage class it belongs to."
    )
    st.write(
        "コンビニ飲料の画像をアップロードすると、学習済みの飲料クラスの中から、"
        "最も可能性が高い商品カテゴリを予測します。"
    )

    with st.expander("Model information / モデル情報"):
        st.write("**Model architecture / モデル構造:** ResNet50")
        st.write("**Training method / 学習方法:** Transfer learning / 転移学習")
        st.write("**Task / タスク:** 11-class beverage image classification / 11クラス飲料画像分類")
        st.write("**Dataset / データセット:** 917 self-collected images / 自分で収集した917枚の画像")
        st.write("**Test accuracy / テスト精度:** 98.66%")
        st.write("**Framework / フレームワーク:** PyTorch, torchvision")
        st.write("**Interface / インターフェース:** Streamlit")
        st.write(
            "**Limitation / 制限事項:** This is a closed-set classifier. "
            "If an uploaded product is not one of the trained 11 classes, "
            "the model will still choose the most similar class."
        )
        st.write(
            "このモデルはクローズドセット分類器です。アップロードされた商品が"
            "学習済みの11クラスに含まれていない場合でも、モデルは最も似ている"
            "クラスを選択します。"
        )

    uploaded_file = st.file_uploader(
        "Upload an image / 画像をアップロード",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded image / アップロード画像", width="stretch")

        with st.spinner("Running inference... / 推論を実行中..."):
            results, device = predict(image)

        top_result = results[0]
        predicted_class = top_result["class_name"]
        confidence = top_result["probability"]

        st.subheader("Prediction Result / 予測結果")

        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(
                "Low confidence prediction. This image may be outside the trained classes "
                "or visually similar to multiple classes.\n\n"
                "信頼度が低い予測です。この画像は学習済みクラスに含まれていない可能性、"
                "または複数のクラスに視覚的に似ている可能性があります。"
            )
            st.write(f"**Top candidate / 最有力候補:** `{predicted_class}`")
        else:
            st.success("Prediction completed. / 予測が完了しました。")
            st.write(f"**Predicted class / 予測クラス:** `{predicted_class}`")

        st.write(f"**Confidence / 信頼度:** `{confidence:.4f}`")
        st.write(f"**Device / 実行デバイス:** `{device}`")

        st.subheader("Top-3 Predictions / Top-3 予測結果")

        for rank, result in enumerate(results, start=1):
            class_name = result["class_name"]
            probability = result["probability"]

            st.write(f"{rank}. `{class_name}` - {probability:.4f}")
            st.progress(float(probability))

    else:
        st.info(
            "Please upload a beverage image to start prediction.\n\n"
            "予測を開始するには、飲料画像をアップロードしてください。"
        )


if __name__ == "__main__":
    main()
