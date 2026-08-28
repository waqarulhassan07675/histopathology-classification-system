import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# Dataset path
dataset_path = r"dataset/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"

# Collect all PNG images
image_files = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(".png"):
            image_files.append(os.path.join(root, file))

print(f"Total images found: {len(image_files)}")

# Select 6 random images
sample_images = random.sample(image_files, 6)

plt.figure(figsize=(12, 8))

for i, image_path in enumerate(sample_images):
    img = Image.open(image_path)

    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.axis("off")

    if "benign" in image_path.lower():
        plt.title("Benign")
    else:
        plt.title("Malignant")

plt.tight_layout()

plt.savefig("outputs/graphs/dataset_samples.png", dpi=300)

plt.show()