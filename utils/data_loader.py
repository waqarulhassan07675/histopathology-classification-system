import os
from PIL import Image

def load_image_paths(dataset_path):
    """
    Load image file paths and labels from the BreaKHis dataset.

    Returns:
        image_paths (list)
        labels (list)
    """

    image_paths = []
    labels = []

    for category in ["benign", "malignant"]:

        category_path = os.path.join(dataset_path, category)

        for root, dirs, files in os.walk(category_path):

            for file in files:

                if file.lower().endswith(".png"):

                    image_paths.append(
                        os.path.join(root, file)
                    )

                    labels.append(category)

    return image_paths, labels