from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_project_files_exist():
    required_files = [
        "README.md",
        "requirements.txt",
        "split_dataset.py",
        "check_dataset.py",
        "train_beverage_resnet50.py",
        "evaluate_beverage_resnet50.py",
        "predict_one.py",
        "streamlit_app.py",
    ]

    for file_path in required_files:
        assert (PROJECT_ROOT / file_path).exists(), f"Missing file: {file_path}"


def test_required_log_files_exist():
    required_files = [
        "logs/class_names.json",
        "logs/confusion_matrix.csv",
        "logs/test_metrics.txt",
    ]

    for file_path in required_files:
        assert (PROJECT_ROOT / file_path).exists(), f"Missing file: {file_path}"


def test_class_names_count():
    class_names_path = PROJECT_ROOT / "logs" / "class_names.json"

    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    assert isinstance(class_names, list)
    assert len(class_names) == 11


def test_demo_gif_exists():
    gif_path = PROJECT_ROOT / "assets" / "beverage_classification_demo.gif"
    assert gif_path.exists(), "Demo GIF is missing"
