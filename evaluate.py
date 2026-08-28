import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

from tensorflow.keras.models import load_model

from utils.data_loader import load_image_paths
from utils.dataset_split import split_dataset
from utils.data_generator import create_dataset


# =====================================================
# Create Output Folders
# =====================================================

os.makedirs("outputs/confusion_matrix", exist_ok=True)
os.makedirs("outputs/roc", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

# =====================================================
# Load Model
# =====================================================

print("=" * 50)
print("Loading Model...")
print("=" * 50)

model = load_model("outputs/models/vgg16_best.keras")

print("Model Loaded Successfully")

# =====================================================
# Load Dataset
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
# Create Test Dataset
# =====================================================

test_dataset = create_dataset(
    test_images,
    test_labels,
    shuffle=False,
)

print()
print("=" * 50)
print("Test Dataset Ready")
print("=" * 50)

print(f"Test Images : {len(test_images)}")

# =====================================================
# Predictions
# =====================================================

print()
print("=" * 50)
print("Generating Predictions...")
print("=" * 50)

predictions = model.predict(test_dataset)

y_pred = (predictions > 0.5).astype(int).flatten()

y_true = np.array(test_labels)

# Convert labels if stored as text
if isinstance(y_true[0], str):
    y_true = np.array(
        [1 if label == "malignant" else 0 for label in y_true]
    )

# =====================================================
# Accuracy
# =====================================================

accuracy = np.mean(y_pred == y_true)

print()
print("=" * 50)
print("Evaluation Results")
print("=" * 50)

print(f"Test Accuracy : {accuracy * 100:.2f}%")

# =====================================================
# Classification Report
# =====================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=["Benign", "Malignant"],
)

print()
print(report)

with open(
    "outputs/reports/classification_report.txt",
    "w",
) as f:
    f.write(report)

# =====================================================
# Confusion Matrix
# =====================================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))

plt.imshow(cm, interpolation="nearest")

plt.title("Confusion Matrix")

plt.colorbar()

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
        )

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix/confusion_matrix.png"
)

plt.close()

# =====================================================
# ROC Curve
# =====================================================

fpr, tpr, _ = roc_curve(
    y_true,
    predictions.flatten()
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig(
    "outputs/roc/roc_curve.png"
)

plt.close()

print()
print("=" * 50)
print("Evaluation Complete")
print("=" * 50)

print("Confusion Matrix Saved")
print("ROC Curve Saved")
print("Classification Report Saved")