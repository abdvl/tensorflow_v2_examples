import time
import numpy as np
import tensorflow as tf

print(tf.__version__)

# Loading the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Image normalization
# can use tf.keras.layers.Flatten instead
X_train = X_train / 255.
X_test = X_test / 255.
print(X_train.shape)

X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)
print(X_train.shape)


# Defining normal (non distributed) model
model_normal = tf.keras.models.Sequential()
model_normal.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784,)))
model_normal.add(tf.keras.layers.Dropout(rate=0.2))
model_normal.add(tf.keras.layers.Dense(units=10, activation='softmax'))
model_normal.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])


# Defining the Distributed Strategy
distribute = tf.distribute.MirroredStrategy()
# Defining a distributed model
with distribute.scope():
  model_distributed = tf.keras.models.Sequential()
  model_distributed.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784,)))
  model_distributed.add(tf.keras.layers.Dropout(rate=0.2))
  model_distributed.add(tf.keras.layers.Dense(units=10, activation='softmax'))
  model_distributed.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])


# Speed comparison between normal training and distributed training process
start_time = time.time()
model_distributed.fit(X_train, y_train, epochs=10, batch_size=25)
print("Distributed training took: {}".format(time.time() - start_time))


start_time = time.time()
model_normal.fit(X_train, y_train, epochs=10, batch_size=25)
print("Normal training took: {}".format(time.time() - start_time))
