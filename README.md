# Chest X-Ray Pneumonia Detector

🏥 AI-powered Pneumonia Detection from Chest X-Rays using VGG16 Transfer Learning  
98.5% Recall · Streamlit App · Deployed on Hugging Face

Live demo: https://huggingface.co/spaces/FIZA255/pneumonia-detector

---

## Table of contents
- [About](#about)
- [Key Features](#key-features)
- [Model & Results](#model--results)
- [Demo](#demo)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Quickstart — Run the Streamlit app locally](#quickstart---run-the-streamlit-app-locally)
  - [Run inference from the command line (example)](#run-inference-from-the-command-line-example)
- [Training (Transfer Learning)](#training-transfer-learning)
- [Dataset](#dataset)
- [Project structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## About
This repository contains code, model artifacts, and a Streamlit user interface for detecting pneumonia in chest X-ray images using transfer learning with the VGG16 architecture. The model is fine-tuned to achieve high recall for pneumonia detection and is exposed via a lightweight web app for demo and rapid testing.

## Key Features
- VGG16-based transfer learning for robust feature extraction
- High recall for pneumonia detection (reported: 98.5%)
- Interactive Streamlit web app for image upload and prediction
- Easily reproducible training and inference pipeline
- Deployed demo on Hugging Face Spaces

## Model & Results
- Base model: VGG16 (pre-trained on ImageNet) with custom classification head
- Fine-tuned with class-weighting / augmentation to prioritize recall
- Reported performance: 98.5% recall on the validation/test split used during development
  - Note: please refer to the `notebooks/` or `reports/` folder (if present) for full evaluation details, confusion matrix, and other metrics.

## Demo
Try the live demo on Hugging Face Spaces:
https://huggingface.co/spaces/FIZA255/pneumonia-detector

## Getting Started

### Requirements
- Python 3.8+
- Recommended: GPU for training (CUDA) but not required for inference
- Common Python dependencies (example):
  - tensorflow or torch (depending on implementation)
  - numpy, pandas, scikit-learn
  - pillow, opencv-python
  - streamlit

Install dependencies (example):
```bash
git clone https://github.com/FizaAslam1/Chest-XRay-Pneumonia-Detector.git
cd Chest-XRay-Pneumonia-Detector
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
If there is no `requirements.txt`, create one or install packages used in the repo, e.g.:
```bash
pip install streamlit tensorflow numpy pandas scikit-learn pillow opencv-python
```

### Quickstart — Run the Streamlit app locally
(Replace `app.py` below with the Streamlit script name in this repo if different.)
```bash
streamlit run app.py
```
Open the shown local URL in your browser, upload a chest X-ray image, and view prediction and metadata.

### Run inference from the command line (example)
If the repo includes an inference script (e.g., `infer.py`):
```bash
python infer.py --image path/to/xray.jpg --model path/to/model.h5
```
Adjust flags to match the actual CLI in this repo.

## Training (Transfer Learning)
High-level steps to reproduce training:
1. Prepare dataset folders (train/val/test) with class subfolders (e.g., `NORMAL`, `PNEUMONIA`).
2. Use data augmentation (rotation, flips, brightness) to reduce overfitting.
3. Load VGG16 (include_top=False, weights='imagenet'), add a global pooling layer, dropout, and dense output layer for binary classification.
4. Freeze convolutional base initially, train the top layers, then unfreeze selected layers and fine-tune with a low learning rate.
5. Use callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau.
6. Evaluate on hold-out test set and generate confusion matrix, ROC curve, and per-class precision/recall.

Example training snippet (conceptual):
```python
base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Dropout(0.5)(x)
out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=base.input, outputs=out)
# compile, train, fine-tune...
```

## Dataset
This project commonly uses the public "Chest X-Ray Images (Pneumonia)" dataset (Kermany et al.), available on Kaggle and other sources. Ensure you have the right permissions and adhere to the dataset's license and terms when using or redistributing.

If you used a different dataset, document its source and license in this README or a separate DATASET.md.

## Project structure (suggested)
If the repository follows typical layout, you might see:
```
.
├── app.py                 # Streamlit app
├── requirements.txt
├── train.py               # Training script
├── infer.py               # Inference script
├── models/                # Saved model weights
├── data/                  # Dataset or dataset pointers (not usually committed)
├── notebooks/             # EDA and training experiments
├── README.md
└── LICENSE
```
Update this section to reflect the actual files in this repo.

## Contributing
Contributions are welcome. To contribute:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes and push: `git push origin feature/your-feature`
4. Create a pull request describing your changes

Please include tests or reproducible instructions for significant changes.

## License
Specify a license for the project (e.g., MIT, Apache-2.0). If you don't have one yet, add a `LICENSE` file and update this section. Example placeholder:
```
MIT License
```

## Acknowledgements
- The Chest X-Ray Pneumonia dataset and its authors (if used)
- TensorFlow/Keras or PyTorch communities for model building blocks
- Streamlit and Hugging Face for deployment tools and hosting

## Contact
Project maintainer: FizaAslam1  
For questions, issues, or collaboration: open an issue on GitHub or contact via GitHub profile.

---
If you'd like, I can:
- Commit this README.md directly to the repository (I will confirm repo and branch details before writing), or
- Tailor the README to include exact filenames/commands after I inspect the repo contents and training/inference scripts.
