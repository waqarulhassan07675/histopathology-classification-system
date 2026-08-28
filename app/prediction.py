import os
import time
import cv2
import joblib
import numpy as np

from tensorflow.keras.models import load_model


# =====================================================
# Model Paths
# =====================================================

MODEL_PATHS = {

    "VGG16":
        "outputs/models/vgg16_best.keras",

    "ResNet50":
        "outputs/models/resnet50_best.keras",

    "DenseNet201":
        "outputs/models/densenet201_best.keras",

    "MobileNetV2":
        "outputs/models/mobilenet_best.keras",

    "Linear SVM":
        "outputs/models/svm_best.pkl",

}


# =====================================================
# Cache Models
# =====================================================

_loaded_models = {}


# =====================================================
# Load Model
# =====================================================

def load_selected_model(model_name):

    if model_name in _loaded_models:
        return _loaded_models[model_name]

    path = MODEL_PATHS[model_name]

    if model_name == "Linear SVM":

        model = joblib.load(path)

    else:

        model = load_model(path)

    _loaded_models[model_name] = model

    return model


# =====================================================
# Read Uploaded Image
# =====================================================

def read_uploaded_image(uploaded_file):

    uploaded_file.seek(0)

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR,
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    return image


# =====================================================
# Deep Learning Preprocessing
# =====================================================

def preprocess_dl(image):

    img = cv2.resize(image, (224, 224))

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    return img


# =====================================================
# SVM Preprocessing
# =====================================================

def preprocess_svm(image):

    img = cv2.resize(image, (32, 32))

    img = img.astype("float32") / 255.0

    img = img.flatten()

    img = img.reshape(1, -1)

    return img


# =====================================================
# Predict
# =====================================================

def predict(uploaded_file, model_name):

    image = read_uploaded_image(uploaded_file)

    model = load_selected_model(model_name)

    start_time = time.time()

    # ------------------------
    # Linear SVM
    # ------------------------

    if model_name == "Linear SVM":

        x = preprocess_svm(image)

        prediction = model.predict(x)[0]

        if hasattr(model, "decision_function"):

            score = float(model.decision_function(x)[0])

            probability = 1 / (1 + np.exp(-score))

        else:

            probability = 0.5

    # ------------------------
    # Deep Learning Models
    # ------------------------

    else:

        x = preprocess_dl(image)

        probability = float(
            model.predict(
                x,
                verbose=0,
            )[0][0]
        )

        prediction = 1 if probability >= 0.5 else 0

    elapsed = time.time() - start_time

    if prediction == 1:

        label = "Malignant"

        confidence = probability

    else:

        label = "Benign"

        confidence = 1 - probability

    return {

        "label": label,

        "confidence": confidence * 100,

        "probability": probability,

        "processing_time": elapsed,

        "image": image,

    }