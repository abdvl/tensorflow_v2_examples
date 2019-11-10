import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb

number_of_words = 20000
max_len = 100

# load dataset
# https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded-when-allow-pickle-false-for-imdb-loa/56062555
# np_load_old = np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=number_of_words)
# np.load = np_load_old

# padding

X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)

#  RNN
model = tf.keras.Sequential()

model.add(tf.keras.layers.Embedding(input_dim=number_of_words, output_dim=128, input_shape=(X_train.shape[1],)))

# tf.keras.layers.CuDNNLSTM
model.add(tf.keras.layers.LSTM(units=128, activation='tanh'))

model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# print model
print(model.summary())

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3, batch_size=128)

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test accuracy: {}".format(test_accuracy))

# Saving the architecture (topology) of the network
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")


