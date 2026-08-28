from tensorflow.keras.utils import plot_model
from models.vgg16_model import build_vgg16

model = build_vgg16()

plot_model(
    model,
    to_file="outputs/graphs/vgg16_architecture.png",
    show_shapes=True,
    show_layer_names=True
)

print("Saved")