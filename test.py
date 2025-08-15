import numpy as np
import tensorflow as tf
from PIL import Image
import sys

# Load the trained CNN model
model = tf.keras.models.load_model("digit_mlp_classifier.h5")

if len(sys.argv) < 2:
    print("Usage: python test.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

# Load and preprocess the image
img = Image.open(image_path).convert('L')  # grayscale
img = img.resize((28, 28))  # resize to 28x28

img_array = np.array(img)
img_array = 255 - img_array  # invert if your digits are black-on-white
img_array = img_array.astype("float32") / 255.0  # normalize

# Add channel dimension for CNN: (28,28,1)
img_array = np.expand_dims(img_array, axis=-1)
# Add batch dimension: (1,28,28,1)
img_array = np.expand_dims(img_array, axis=0)

pred_probability = model.predict(img_array)
pred_label = np.argmax(pred_probability)

print(f"Predicted label: {pred_label}")
