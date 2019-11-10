import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook
from tensorflow.keras.preprocessing.image import ImageDataGenerator


dataset_path_new = "./cats_and_dogs_filtered/"
train_dir = os.path.join(dataset_path_new, "train")
validation_dir = os.path.join(dataset_path_new, "validation")

# Loading the pre-trained model (MobileNetV2)
IMG_SHAPE = (128, 128, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
print(base_model.summary())

# Freezing the base model
base_model.trainable = False

# Defining the custom head for our network
print(base_model.output)

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

prediction_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')(global_average_layer)

# Defining the model

model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction_layer)

# print model
print(model.summary())

# RMSprop is best for mobileNet
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# Creating Data Generators
data_gen_train = ImageDataGenerator(rescale=1/255.)
data_gen_valid = ImageDataGenerator(rescale=1/255.)


train_generator = data_gen_train.flow_from_directory(train_dir, target_size=(128,128), batch_size=128, class_mode="binary")
valid_generator = data_gen_valid.flow_from_directory(validation_dir, target_size=(128,128), batch_size=128, class_mode="binary")


# Training the model
model.fit_generator(train_generator, epochs=5, validation_data=valid_generator)

# model.fit(X_train, y_train, epochs=5)


valid_loss, valid_accuracy = model.evaluate_generator(valid_generator)
# test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Valid accuracy: {}".format(valid_accuracy))


# Fine tuning
base_model.trainable = True
print("Number of layers in the base model: {}".format(len(base_model.layers)))
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Compiling the model for fine-tuning
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())
model.fit_generator(train_generator,  epochs=5, validation_data=valid_generator)

valid_loss, valid_accuracy = model.evaluate_generator(valid_generator)

print("Valid accuracy: {}".format(valid_accuracy))



# Saving the architecture (topology) of the network
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")


