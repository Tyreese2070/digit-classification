import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print("TensorFlow version: ", tf.__version__)

# load data set
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# flatten the images, and normalise pixel values (0,1)
x_train_flat = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test_flat = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# Linear Classifier
linear_model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)), # Input layer
    tf.keras.layers.Dense(10, activation='softmax') # Output Layer, 10 classes for digits 0-9
])

linear_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

linear_model.fit(x_train_flat, y_train, batch_size=100, epochs=5)
loss, accuracy = linear_model.evaluate(x_test_flat, y_test, verbose=0)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

# Multi layer pereceptron
x_train_norm = x_train.astype("float32") / 255.0
x_test_norm = x_test.astype("float32") / 255.0

mlp_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # Flatten the input
    tf.keras.layers.Dense(128, activation='relu'), # Hidden layer 1 (128 neurons)
    tf.keras.layers.Dense(128, activation='relu'), # Hidden layer 2 (128 neurons)
    tf.keras.layers.Dense(10, activation='softmax') # Output layer (10 classes for 0-9)
])

mlp_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

mlp_model.fit(x_train_norm, y_train, epochs=3)
loss2, accuracy2 = mlp_model.evaluate(x_test_norm, y_test, verbose=0)
print(f"Test loss: {loss2:.4f}")
print(f"Test accuracy: {accuracy2:.4f}")