import tensorflow as tf
from utils.preprocessing import preprocess_image


def create_dataset(image_paths, labels, batch_size=32, shuffle=True):
    """
    Create a TensorFlow Dataset.

    Parameters
    ----------
    image_paths : list
    labels : list
    batch_size : int
    shuffle : bool

    Returns
    -------
    tf.data.Dataset
    """

    # Convert labels to binary values
    binary_labels = [
        0 if label == "benign" else 1
        for label in labels
    ]

    def load_data(path, label):

        image = tf.numpy_function(
            preprocess_image,
            [path],
            tf.float32
        )

        image.set_shape((224, 224, 3))

        return image, label

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, binary_labels)
    )

    dataset = dataset.map(
        load_data,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset