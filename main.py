import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print("TensorFlow version: ", tf.__version__)

# load data set
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# flatten the images, and normalise pixel values (0,1)
x_train_flat = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test_flat = x_test.reshape(-1, 28*28).astype("float32") / 255.0