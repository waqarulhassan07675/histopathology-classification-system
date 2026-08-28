from utils.data_loader import load_image_paths
from utils.dataset_split import split_dataset

dataset_path = r"dataset/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"

images, labels = load_image_paths(dataset_path)

(
    train_images,
    val_images,
    test_images,
    train_labels,
    val_labels,
    test_labels,
) = split_dataset(images, labels)

print("=" * 45)
print("Dataset Split Summary")
print("=" * 45)

print(f"Total Images      : {len(images)}")
print(f"Training Images   : {len(train_images)}")
print(f"Validation Images : {len(val_images)}")
print(f"Testing Images    : {len(test_images)}")