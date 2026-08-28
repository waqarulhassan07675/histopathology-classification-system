from utils.data_loader import load_image_paths

dataset_path = r"dataset/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"

images, labels = load_image_paths(dataset_path)

print("="*40)
print("Dataset Successfully Loaded")
print("="*40)

print(f"Total Images : {len(images)}")
print(f"Total Labels : {len(labels)}")

print()

print("First Image")

print(images[0])

print()

print("First Label")

print(labels[0])