import tensorflow_core as tf
import numpy as np

from tensorflow_core.python.keras.datasets import fashion_mnist


print(tf.__version__)

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Image normalization
# can use tf.keras.layers.Flatten instead
X_train = X_train / 255.
X_test = X_test / 255.
print(X_train.shape)

X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)
print(X_train.shape)


# Building a model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784,)))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

model.fit(X_train, y_train, epochs=1)
# Evaluating the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_accuracy))



# Converting the model into TensorFlow Lite version

# model_name = 'mobile_model.h5'
# tf.keras.models.save_model(model, model_name)

# Creating the TFLite Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("tf_model.tflite", "wb") as f:
  f.write(tflite_model)





