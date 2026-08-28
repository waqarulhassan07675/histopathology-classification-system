from utils.data_loader import load_image_paths
from utils.preprocessing import preprocess_image

dataset_path = r"dataset/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"

images, labels = load_image_paths(dataset_path)

sample_image = preprocess_image(images[0])

print("="*40)
print("Image Preprocessing Test")
print("="*40)

print("Shape :", sample_image.shape)

print("Datatype :", sample_image.dtype)

print("Minimum Pixel :", sample_image.min())

print("Maximum Pixel :", sample_image.max())