# Breast Cancer Histopathology Classification System

## MSc Computer Science and Technology Project

An AI-based breast cancer histopathology image classification system developed as part of an MSc Computer Science and Technology project at Ulster University.

The system investigates and compares deep learning and traditional machine learning approaches for classifying breast histopathology images into **Benign** and **Malignant** categories.

---

## Project Overview

Breast cancer diagnosis through histopathological examination requires the analysis of tissue images by experienced medical professionals. This project investigates the use of machine learning and deep learning techniques to support automated classification of histopathological images.

Five classification approaches were implemented and evaluated:

- VGG16
- ResNet50
- DenseNet201
- MobileNetV2
- Linear Support Vector Machine (SVM)

The best-performing deep learning model was subsequently integrated into an interactive Streamlit-based application for image classification.

---

## Objectives

The main objectives of the project are:

1. To investigate automated breast cancer classification using histopathological images.
2. To prepare and preprocess the BreaKHis dataset for machine learning.
3. To implement transfer learning using VGG16, ResNet50, DenseNet201 and MobileNetV2.
4. To implement Linear SVM as a traditional machine learning baseline.
5. To evaluate and compare the classification performance of the five models.
6. To develop an interactive web-based classification application.
7. To demonstrate the practical deployment of the selected model.

---

## Dataset

The project uses the **BreaKHis (Breast Cancer Histopathological Database)** dataset.

The dataset contains histopathological images of breast tissue belonging to benign and malignant categories.

The dataset itself is **not included in this repository** because of its size. Users wishing to reproduce the experiments should obtain the dataset from its original source and configure the local dataset path according to the project code.

---

## Models

### Deep Learning Models

The following pre-trained CNN architectures were investigated using transfer learning:

| Model | Approach |
|---|---|
| VGG16 | Transfer Learning |
| ResNet50 | Transfer Learning |
| DenseNet201 | Transfer Learning |
| MobileNetV2 | Transfer Learning |

### Traditional Machine Learning

A **Linear Support Vector Machine (SVM)** was implemented as a baseline machine learning approach.

---

## Model Performance

The experimental comparison produced the following test accuracy results:

| Model | Accuracy |
|---|---:|
| DenseNet201 | **93.09%** |
| VGG16 | 87.11% |
| MobileNetV2 | 86.52% |
| ResNet50 | 83.49% |
| Linear SVM | 72.70% |

Among the evaluated approaches, **DenseNet201 achieved the highest classification accuracy of 93.09%**.

The results are used in the project to compare the effectiveness of deep transfer learning architectures against the traditional SVM baseline.

---

## Application

The selected model is integrated into a **Streamlit web application**.

The application provides:

- Histopathology image upload
- Image preview
- Model selection
- Benign/Malignant prediction
- Prediction confidence
- Processing time
- Image size information
- Model performance comparison
- Interactive visualisation of model results

The application is intended as a **research prototype and computer-aided classification demonstration** and is not intended to replace professional medical diagnosis.

---

## Application Workflow

```text
Histopathology Image
        |
        v
Image Upload
        |
        v
Preprocessing
        |
        v
Selected Classification Model
        |
        v
Prediction
        |
        +------------------+
        |                  |
        v                  v
     Benign             Malignant
        |
        v
Confidence & Processing Information
Project Structure
BreastCancerProject/
│
├── app/
│   ├── __init__.py
│   ├── charts.py
│   ├── prediction.py
│   └── styles.py
│
├── notebooks/
│
├── utils/
│   ├── app.py
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
├── evaluate.py
├── evaluate_densenet201.py
├── evaluate_mobilenet.py
├── evaluate_resnet50.py
├── evaluate_svm.py
├── model_diagram.py
│
├── train.py
├── train_densenet201.py
├── train_mobilenet.py
├── train_resnet50.py
├── train_svm.py
│
├── test_*.py
│
├── requirements.txt
├── .gitignore
└── README.md
Installation

Clone the repository:

git clone https://github.com/waqarulhassan07675/histopathology-classification-system.git

Move into the project directory:

cd histopathology-classification-system

Create and activate a Python virtual environment:

Windows
python -m venv venv
venv\Scripts\activate

Install the required dependencies:

pip install -r requirements.txt
Running the Application

After configuring the dataset and required model files, start the Streamlit application:

streamlit run dashboard.py

The application will open in a web browser.

Model Evaluation

The repository contains separate scripts for training and evaluating the implemented models.

Evaluation scripts include:

evaluate.py
evaluate_densenet201.py
evaluate_mobilenet.py
evaluate_resnet50.py
evaluate_svm.py

The project evaluates model performance using measures including:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
ROC Curve
Testing

Testing scripts are included to verify important components of the system, including:

Data loading
Image preprocessing
Data generation
Dataset splitting
Data augmentation
Individual model functionality
Reproducibility

The repository contains the source code used to implement the classification pipeline and Streamlit application.

The following are intentionally excluded from the repository:

BreaKHis dataset
Trained model files
Generated output files
Python virtual environment
Temporary/cache files

These files can be generated or configured locally when reproducing the experiments.

Limitations

This project represents an academic research prototype rather than a clinically validated diagnostic system.

The classification results should therefore not be interpreted as a medical diagnosis. Further validation using larger and more diverse datasets, external clinical data and prospective clinical evaluation would be required before any real-world clinical application.

Academic Context

Degree: MSc Computer Science and Technology
Institution: Ulster University
Project: Breast Cancer Histopathological Classification System

Author: Waqar Ul Hassan