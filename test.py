import numpy as np
import tensorflow as tf
from PIL import Image
import sys

model = tf.keras.models.load_model("digit_mlp_classifier.h5")

if len(sys.argv) < 2:
    print("Usage: python test.py <image_path>")
    sys.exit(1)
    
image_path = sys.argv[1]

img = Image.open(image_path).convert('L')  # Convert to grayscale
img = img.resize((28, 28))  # Resize to 28x28

img_array = np.array(img)
img_array = 255-img_array  # Invert colors
img_array = img_array.astype("float32") / 255.0  # Normalise pixel values

img_array = np.expand_dims(img_array, axis=0)  # shape (1,28,28)

pred_probability = model.predict(img_array)
pred_label = np.argmax(pred_probability)

print(f"Predicted label: {pred_label}"  )