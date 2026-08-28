import os
import pickle

import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from utils.data_loader import load_image_paths
from utils.dataset_split import split_dataset
from utils.data_generator import create_dataset

from models.densenet201_model import build_densenet201


# =====================================================
# Dataset Path
# =====================================================

DATASET_PATH = r"dataset/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"

# =====================================================
# Output Folders
# =====================================================

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/graphs", exist_ok=True)
os.makedirs("outputs/history", exist_ok=True)

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

train_dataset = create_dataset(train_images, train_labels)

validation_dataset = create_dataset(
    val_images,
    val_labels,
    shuffle=False,
)

print("Dataset Loaded Successfully")

print()

print("=" * 60)
print("Building DenseNet201 Model...")
print("=" * 60)

model = build_densenet201()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("Model Compiled Successfully")

callbacks = [

    ModelCheckpoint(
        "outputs/models/densenet201_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
    ),

]

print()

print("=" * 60)
print("Training Started...")
print("=" * 60)

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=10,

    callbacks=callbacks,

)

print()

print("=" * 60)
print("Training Finished")
print("=" * 60)

# =====================================================
# Save History
# =====================================================

with open(
    "outputs/history/densenet201_history.pkl",
    "wb",
) as f:

    pickle.dump(history.history, f)

# =====================================================
# Accuracy Graph
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training")

plt.plot(history.history["val_accuracy"], label="Validation")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("DenseNet201 Accuracy")

plt.legend()

plt.savefig(
    "outputs/graphs/densenet201_accuracy.png"
)

plt.close()

# =====================================================
# Loss Graph
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training")

plt.plot(history.history["val_loss"], label="Validation")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("DenseNet201 Loss")

plt.legend()

plt.savefig(
    "outputs/graphs/densenet201_loss.png"
)

plt.close()

print()

print("Training graphs saved successfully.")
print("Best model saved successfully.")