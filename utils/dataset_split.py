from sklearn.model_selection import train_test_split


def split_dataset(image_paths, labels):
    """
    Split dataset into:
    70% Training
    15% Validation
    15% Testing
    """

    # Train (70%) / Temp (30%)
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels
    )

    # Validation (15%) / Test (15%)
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images,
        temp_labels,
        test_size=0.50,
        random_state=42,
        stratify=temp_labels
    )

    return (
        train_images,
        val_images,
        test_images,
        train_labels,
        val_labels,
        test_labels,
    )