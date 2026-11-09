🫁 Chest X-Ray Pneumonia Detector

Deep learning app that classifies chest X-rays as NORMAL or PNEUMONIA using VGG16 transfer learning, tuned to minimise missed pneumonia cases (high recall).

🔗 Live demo: Hugging Face Space (built and deployed by me)

⚠️ Disclaimer: Research and demo project only. It is not a medical device and must not be used for diagnosis.


Results (held-out test set, 624 images)
Metric	Score
Recall (Pneumonia)	98.5%
Precision	80.3%
F1 score	88.5%
Accuracy	84.0%

The model is deliberately recall-focused: in screening, a missed pneumonia case costs more than a false alarm. The trade-off is lower precision (more false positives). Validation accuracy during training was about 95%, and the gap to test accuracy is discussed under Limitations.

Approach
Dataset: Chest X-Ray Images (Pneumonia), Kermany et al. (Kaggle). Split: 4,173 train / 1,043 validation / 624 test.
Model: VGG16 (ImageNet weights, frozen base) → GlobalAveragePooling → Dense(256) → Dropout(0.5) → Dense(128) → Dropout(0.3) → Sigmoid. Input 224×224×3.
Imbalance handling: class weights (NORMAL ≈ 1.94, PNEUMONIA ≈ 0.67).
Augmentation: rotation (±10°), horizontal flip.
Training: Adam (lr = 1e-4), binary cross-entropy, EarlyStopping and ReduceLROnPlateau (best weights restored from epoch 15).
Run locally
bash
git clone https://github.com/FizaAslam1/Chest-XRay-Pneumonia-Detector.git
cd Chest-XRay-Pneumonia-Detector
pip install -r requirements.txt
streamlit run app.py

Model file: pneumonia_detector.keras (see the Hugging Face Space for the deployed weights).

Repository structure
├── app.py                    # Streamlit app
├── class_info.json           # class labels
├── pneumonia_cnn.ipynb       # training and evaluation notebook
├── requirements.txt
└── README.md
Limitations and next steps
Test accuracy (84%) is lower than validation accuracy (~95%), which suggests a distribution shift between the dataset's train/val and test sets. Next: cross-validation and a larger external test set.
Add Grad-CAM to show which lung regions drive each prediction.
Fine-tune the top VGG16 blocks and compare with EfficientNet/ResNet.
Add ROC-AUC and a confusion matrix to the report.
Tech stack

Python · TensorFlow/Keras · VGG16 · Streamlit · Hugging Face Spaces

Author

Fiza Aslam, Data Scientist and ML Engineer · LinkedIn · GitHub
