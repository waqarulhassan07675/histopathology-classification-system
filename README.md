# Breast Cancer Histopathology Classification System

## MSc Computer Science Project

An AI-based breast cancer histopathology image classification system developed using deep learning and traditional machine learning techniques.

The system evaluates four transfer learning architectures and one traditional machine learning classifier for benign and malignant breast tissue classification.

## Models

The project evaluates:

- VGG16
- ResNet50
- DenseNet201
- MobileNetV2
- Linear SVM

## Dataset

The system uses the BreaKHis breast histopathological image dataset.

The dataset is not included in this repository because of its size.

## Main Features

- Histopathology image upload
- Image preview
- Image preprocessing
- Benign/Malignant classification
- Confidence estimation
- Model selection
- Five-model comparison
- Confusion matrix evaluation
- ROC curve evaluation
- Accuracy comparison
- Interactive Streamlit dashboard

## Best Performing Model

DenseNet201 achieved the highest test accuracy among the evaluated models.

## Project Structure

```text
BreastCancerProject/
│
├── app/
│   ├── charts.py
│   ├── prediction.py
│   └── styles.py
│
├── notebooks/
│
├── utils/
│   ├── augmentation.py
│   ├── data_generator.py
│   ├── data_loader.py
│   ├── dataset_info.py
│   ├── dataset_split.py
│   ├── evaluate.py
│   ├── preprocessing.py
│   ├── show_images.py
│   └── train.py
│
├── dashboard.py
├── evaluate_densenet201.py
├── evaluate_mobilenet.py
├── evaluate_resnet50.py
├── requirements.txt
└── README.md