from utils.data_loader import load_image_paths
from utils.dataset_split import split_dataset
from utils.data_generator import create_dataset

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

train_dataset = create_dataset(
    train_images,
    train_labels
)

for images, labels in train_dataset.take(1):

    print("=" * 40)
    print("Training Batch")
    print("=" * 40)

    print("Images Shape :", images.shape)
    print("Labels Shape :", labels.shape)