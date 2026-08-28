import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

from utils.data_loader import load_image_paths
from utils.augmentation import augmentation

# Dataset path
dataset_path = r"dataset/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"

# Load all image paths
image_paths, labels = load_image_paths(dataset_path)

# Select one random image
img_path = random.choice(image_paths)

print("Selected Image:")
print(img_path)

# Load image
img = image.load_img(img_path, target_size=(224, 224))

# Convert to array
x = image.img_to_array(img)

# Reshape
x = x.reshape((1,) + x.shape)

# Display augmented images
plt.figure(figsize=(12,8))

i = 1

for batch in augmentation.flow(x, batch_size=1):

    plt.subplot(2,3,i)

    plt.imshow(batch[0].astype("uint8"))

    plt.axis("off")

    i += 1

    if i > 6:
        break

plt.tight_layout()
plt.savefig("outputs/graphs/data_augmentation.png", dpi=300)
plt.show()