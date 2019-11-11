from tensorflow.keras.datasets import fashion_mnist
import imageio


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
for i in range(5):
    imageio.imwrite("uploads/{0}.jpg".format(i), X_test[i])



