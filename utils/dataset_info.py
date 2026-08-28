import os

# Dataset path
dataset_path = r"dataset/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"

print("=" * 50)
print("Breast Cancer Dataset Information")
print("=" * 50)

for category in os.listdir(dataset_path):

    category_path = os.path.join(dataset_path, category)

    # Skip files (README, .sh etc.)
    if not os.path.isdir(category_path):
        continue

    total_images = 0

    for root, dirs, files in os.walk(category_path):
        total_images += len(
            [file for file in files if file.lower().endswith(".png")]
        )

    print(f"{category}: {total_images} images")