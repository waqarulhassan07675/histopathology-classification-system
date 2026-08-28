import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from utils.data_loader import load_image_paths
from utils.dataset_split import split_dataset


# =====================================================
# Dataset Path
# =====================================================

DATASET_PATH = r"dataset/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/graphs", exist_ok=True)


print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

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
# Load Images
# =====================================================

def load_images(image_paths):

    X = []

    for path in image_paths:

        img = cv2.imread(path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Smaller size for faster training
        img = cv2.resize(img, (32, 32))

        img = img.astype("float32") / 255.0

        X.append(img.flatten())

    return np.array(X)


print()
print("Loading Images...")

X_train = load_images(train_images)
X_val = load_images(val_images)

y_train = np.array(
    [0 if x == "benign" else 1 for x in train_labels]
)

y_val = np.array(
    [0 if x == "benign" else 1 for x in val_labels]
)


print("Training Samples :", len(X_train))
print("Validation Samples :", len(X_val))


# =====================================================
# Train Model
# =====================================================

print()
print("=" * 60)
print("Training Linear SVM...")
print("=" * 60)

model = LinearSVC(
    random_state=42,
    max_iter=5000
)

model.fit(X_train, y_train)

print()
print("Training Finished")


# =====================================================
# Validation Accuracy
# =====================================================

pred = model.predict(X_val)

accuracy = accuracy_score(y_val, pred)

print(f"Validation Accuracy : {accuracy*100:.2f}%")

# =====================================================
# Save Model
# =====================================================

with open(
    "outputs/models/svm_best.pkl",
    "wb",
) as f:

    pickle.dump(model, f)

print("Model Saved")


# =====================================================
# Accuracy Graph
# =====================================================

plt.figure(figsize=(6,5))

plt.bar(
    ["Linear SVM"],
    [accuracy]
)

plt.ylim(0,1)

plt.ylabel("Accuracy")

plt.title("Linear SVM Validation Accuracy")

plt.savefig(
    "outputs/graphs/svm_accuracy.png"
)

plt.close()

print("Accuracy Graph Saved")