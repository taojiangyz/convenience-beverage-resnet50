# Convenience Store Beverage Image Classification  
# コンビニ飲料画像分類モデル

A ResNet50-based image classification project for recognizing convenience store beverage products from self-photographed images.  
This project covers the full workflow from image data preparation, dataset validation, model training, evaluation, single-image inference, and an interactive Streamlit demo.

ResNet50を用いたコンビニ飲料画像分類プロジェクトです。  
自分で撮影した飲料画像を用いて、データセット作成、データ確認、モデル学習、評価、単一画像推論、Streamlitデモまで一通り実装しました。

---

## Demo

The Streamlit demo allows users to upload a beverage image and view the predicted class, confidence score, and top-3 predictions.

Streamlitデモでは、飲料画像をアップロードすると、予測クラス、信頼度、Top-3予測結果を確認できます。

![Streamlit Beverage Classification Demo](assets/beverage_classification_demo.gif)

---

## Project Overview

This project is designed as a portfolio-level machine learning project.  
The goal is not only to train an image classification model, but also to demonstrate a practical workflow from data preparation to inference demo.

このプロジェクトは、ポートフォリオ用の機械学習プロジェクトとして作成しました。  
単に画像分類モデルを学習するだけでなく、データ準備から推論デモまでの実用的な流れを示すことを目的としています。

### Main Features

- Self-photographed beverage image dataset
- 11-class convenience store beverage classification
- ResNet50 transfer learning using PyTorch
- Dataset validation script
- Model evaluation with classification report and confusion matrix
- Single-image inference script
- English/Japanese Streamlit demo
- Demo GIF for GitHub presentation

### 主な機能

- 自分で撮影した飲料画像データセット
- 11クラスのコンビニ飲料商品分類
- PyTorchによるResNet50の転移学習
- データセット確認スクリプト
- classification report と confusion matrix によるモデル評価
- 単一画像推論スクリプト
- 英日双語のStreamlitデモ
- GitHub表示用のデモGIF

---

## Dataset

The dataset was created from beverage images photographed by myself using a smartphone.  
For the first version, the task was limited to convenience store beverages to keep the scope practical and controllable.

データセットは、スマートフォンを用いて自分で撮影した飲料画像から作成しました。  
初期版では、対象をコンビニ飲料に限定し、実装しやすく管理しやすいタスク設計にしました。

### Dataset Summary

| Item | Value |
|---|---:|
| Total valid images | 917 |
| Number of classes | 11 |
| Training images | 637 |
| Validation images | 131 |
| Test images | 149 |

### Classes

```text
asakatsu_yasai
bihidasu_yogurt
dodekamin
ichinichibun_no_yasai
ocha_lemon_green
oi_ocha_koi_cha
oi_ocha_ryokucha
savas_milk_protein_banana
savas_whey_protein
zero_cider_triple
zero_cider_triple_plus_afa
```

Each class contains approximately 60 to 100 images.  
Images were taken from multiple angles and slightly different distances while keeping the product label visible.

各クラスには約60〜100枚の画像が含まれています。  
商品ラベルが見えるようにしながら、複数の角度と距離から撮影しました。

---

## Model

The model is based on ResNet50 transfer learning.

モデルにはResNet50の転移学習を使用しました。

| Item | Value |
|---|---|
| Model architecture | ResNet50 |
| Framework | PyTorch / torchvision |
| Input size | 224 x 224 |
| Number of output classes | 11 |
| Training method | Transfer learning |
| Loss function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Epochs | 10 |
| Batch size | 16 |

The final fully connected layer of ResNet50 was replaced to match the 11 beverage classes.

ResNet50の最終全結合層を、11クラス分類に合わせて置き換えました。

---

## Evaluation Results

The model was evaluated on the test dataset.

テストデータセットを用いてモデル評価を行いました。

| Metric | Value |
|---|---:|
| Test Accuracy | 98.66% |

Most classes were classified correctly.  
The main confusion occurred between visually similar products, especially:

```text
zero_cider_triple
zero_cider_triple_plus_afa
```

This is reasonable because these two products have very similar packaging.

多くのクラスは正しく分類されました。  
主な誤分類は、パッケージが似ている以下の商品間で発生しました。

```text
zero_cider_triple
zero_cider_triple_plus_afa
```

これは、両商品のパッケージが視覚的に似ているため、自然な誤分類だと考えられます。

Evaluation files are saved in the `logs/` directory:

```text
logs/class_names.json
logs/confusion_matrix.csv
logs/test_metrics.txt
```

---

## Inference

### Single-image inference

Run the following command:

```bash
python3 predict_one.py path/to/image.jpg
```

Example output:

```text
Image: product_dataset/test/oi_ocha_koi_cha/IMG_0541.jpeg
Prediction: oi_ocha_koi_cha
Confidence: 0.7225

Top-3 predictions:
oi_ocha_koi_cha: 0.7225
oi_ocha_ryokucha: 0.0754
savas_whey_protein: 0.0540
```

### Streamlit demo

Run:

```bash
streamlit run streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

The web demo supports English and Japanese.  
Users can upload an image and view the predicted class, confidence score, and top-3 predictions.

Streamlitデモは英日双語に対応しています。  
画像をアップロードすると、予測クラス、信頼度、Top-3予測結果を確認できます。

---

## Project Structure

```text
convenience-beverage-resnet50/
├── assets/
│   └── beverage_classification_demo.gif
├── logs/
│   ├── class_names.json
│   ├── confusion_matrix.csv
│   └── test_metrics.txt
├── check_dataset.py
├── evaluate_beverage_resnet50.py
├── predict_one.py
├── split_dataset.py
├── streamlit_app.py
├── train_beverage_resnet50.py
├── requirements.txt
├── .gitignore
└── README.md
```

Note: The trained model weight file is not included in this repository due to file size.

注意：学習済みモデルの重みファイルは、ファイルサイズの都合によりこのリポジトリには含めていません。

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare dataset

The dataset should be organized in the following format:

```text
product_dataset/
├── raw/
│   ├── class_1/
│   ├── class_2/
│   └── ...
```

Then split the dataset:

```bash
python3 split_dataset.py
```

Check the dataset:

```bash
python3 check_dataset.py
```

### 3. Train the model

```bash
python3 train_beverage_resnet50.py
```

### 4. Evaluate the model

```bash
python3 evaluate_beverage_resnet50.py
```

### 5. Run inference

```bash
python3 predict_one.py path/to/image.jpg
```

### 6. Run Streamlit demo

```bash
streamlit run streamlit_app.py
```

---

## Limitations

This model is a closed-set classifier.

The model can only classify images into the 11 trained beverage classes.  
If an unknown product is uploaded, the model will still choose the most similar class among the trained classes.

このモデルはクローズドセット分類器です。

学習済みの11クラスの中から分類を行うため、未登録の商品がアップロードされた場合でも、最も似ているクラスを選択します。

Possible improvements:

- Add an unknown class
- Apply a confidence threshold
- Collect more images for visually similar products
- Add more beverage categories
- Improve error analysis using Jupyter Notebook

---

## Completed Extensions / 完成済みの拡張

Beyond the basic training and inference workflow, this project includes the following completed extensions:

基本的な学習・推論ワークフローに加えて、本プロジェクトでは以下の拡張も完了しています。

### Jupyter Error Analysis / Jupyterによる誤分類分析

- `notebooks/error_analysis.ipynb`

This notebook loads the confusion matrix and test metrics, visualizes the classification results, and identifies misclassified class pairs. The main observed confusion was between visually similar product packages, especially `zero_cider_triple` and `zero_cider_triple_plus_afa`.

このNotebookでは、confusion matrixとtest metricsを読み込み、分類結果を可視化し、誤分類されたクラスペアを確認します。主な誤分類は、パッケージが非常に似ている `zero_cider_triple` と `zero_cider_triple_plus_afa` の間で発生しました。

### Google Colab GPU Workflow / Google Colab GPU学習ワークフロー

- `notebooks/train_on_colab.ipynb`

This notebook shows how the same training workflow can be moved from a local Mac environment to a cloud GPU environment. It includes Google Drive mounting, dependency installation, CUDA GPU checking, dataset validation, model training, model evaluation, and output file checking.

このNotebookでは、ローカルMac環境で作成した学習ワークフローをGoogle ColabのクラウドGPU環境に移行する流れを示しています。Google Driveのマウント、依存パッケージのインストール、CUDA GPUの確認、データセット確認、モデル学習、モデル評価、出力ファイル確認までを含みます。

Note: The full dataset and trained model weights are not included in this GitHub repository due to file size. To actually run the Colab workflow, the dataset should be placed in Google Drive.

注意：データセット全体と学習済みモデルの重みは、ファイルサイズの都合によりGitHubには含めていません。Colabで実際に実行する場合は、データセットをGoogle Driveに配置する必要があります。

### Basic Tests / 基本テスト

- `tests/test_project_files.py`

Basic pytest tests are included to check whether important project files, evaluation outputs, class definitions, and demo assets exist.

主要なプロジェクトファイル、評価結果、クラス定義、デモ素材が存在するかを確認するための基本的なpytestテストを追加しています。


## Future Work

Possible future improvements include:

- Add image quality checking using OpenCV
- Containerize the environment using Docker or Podman
- Extend inference workflow to AWS or GCP
- Use MLflow for experiment tracking in larger-scale experiments

今後の改善案：

- OpenCVによる画像品質チェック
- DockerまたはPodmanによる環境のコンテナ化
- AWSまたはGCPへの推論ワークフロー拡張
- 大規模実験におけるMLflowによる実験管理

---

## Tech Stack

```text
Python
PyTorch
torchvision
ResNet50
Pillow
scikit-learn
pandas
matplotlib
Streamlit
GitHub
```

---

## Summary

This project demonstrates a complete basic image classification workflow, including data preparation, dataset validation, model training, evaluation, inference, and an interactive demo.

このプロジェクトでは、データ準備、データ確認、モデル学習、評価、推論、インタラクティブデモまで、画像分類タスクの基本的な流れを一通り実装しました。

