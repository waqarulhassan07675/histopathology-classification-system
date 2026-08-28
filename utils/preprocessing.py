import numpy as np
from PIL import Image


IMG_SIZE = (224, 224)


def preprocess_image(image_path):

    if isinstance(image_path, bytes):
        image_path = image_path.decode("utf-8")

    image = Image.open(image_path)

    image = image.convert("RGB")

    image = image.resize(IMG_SIZE)

    image = np.array(image, dtype=np.float32)

    image = image / 255.0

    return image