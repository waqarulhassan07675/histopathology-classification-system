import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    auc,
)

from utils.data_loader import load_image_paths
from utils.dataset_split import split_dataset


# =====================================================
# Output folders
# =====================================================

os.makedirs("outputs/confusion_matrix", exist_ok=True)
os.makedirs("outputs/roc", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)


# =====================================================
# Load Model
# =====================================================

with open("outputs/models/svm_best.pkl", "rb") as f:
    model = pickle.load(f)


# =====================================================
# Dataset
# =====================================================

DATASET_PATH = r"dataset/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"

images, labels = load_image_paths(DATASET_PATH)

(
    train_images,
    val_images,
    test_images,
    train_labels,
    val_labels,
    test_labels,
) = split_dataset(images, labels)


# =====================================================
# Image Loader
# =====================================================

def load_images(image_paths):

    X = []

    for path in image_paths:

        img = cv2.imread(path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (32, 32))

        img = img.astype("float32") / 255.0

        X.append(img.flatten())

    return np.array(X)


print("Loading Test Images...")

X_test = load_images(test_images)

y_true = np.array(
    [0 if x == "benign" else 1 for x in test_labels]
)


# =====================================================
# Prediction
# =====================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_true,
    y_pred
)

print()
print("=" * 50)
print("Evaluation Results")
print("=" * 50)

print(f"Test Accuracy : {accuracy*100:.2f}%")

report = classification_report(
    y_true,
    y_pred,
    target_names=["Benign", "Malignant"]
)

print()
print(report)

with open(
    "outputs/reports/svm_classification_report.txt",
    "w",
) as f:
    f.write(report)


# =====================================================
# Confusion Matrix
# =====================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(6,5))

plt.imshow(cm)

plt.title("SVM Confusion Matrix")

plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i,j]),
            ha="center",
            va="center"
        )

plt.xlabel("Predicted")

plt.ylabel("True")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix/svm_confusion_matrix.png"
)

plt.close()


# =====================================================
# ROC Curve
# =====================================================

scores = model.decision_function(X_test)

fpr, tpr, _ = roc_curve(
    y_true,
    scores
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    "--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("SVM ROC Curve")

plt.legend()

plt.savefig(
    "outputs/roc/svm_roc_curve.png"
)

plt.close()

print()
print("=" * 50)
print("Evaluation Complete")
print("=" * 50)

print("SVM Confusion Matrix Saved")
print("SVM ROC Curve Saved")
print("SVM Classification Report Saved")