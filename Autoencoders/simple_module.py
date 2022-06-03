# loading libraries
from keras.layers import *
from keras import Input, Model
from keras.datasets import mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# load the data
def data_loading(data):
    (X_train, y_train), (X_test, y_test) = data.load_data()
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255
    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

    return X_train, X_test



# train the model
def train_model(model, xtrain, ytrain, xtest, ytest, epochs=15, batch_size=256):
    model.fit(xtrain, ytrain, epochs = epochs, batch_size=batch_size, validation_data=(xtest,ytest))

# plot the result
def plot_result(xtest):
    encoded_input = encoder.predict(xtest)
    decoded_input = decoder.predict(encoded_input)
    plt.figure(figsize=(20, 4))
    for i in range(5):
        # display original image
        ax = plt.subplot(2, 5, i+1)
        plt.imshow(xtest[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstructed image
        ax = plt.subplot(2, 5, i+1+5)
        plt.imshow(decoded_input[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

if __name__ == "__main__":
    x_train, x_test = data_loading(mnist)

    # build the model
    encoding_dim = 15
    input_img = Input(shape=(784,))
    # encoding
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # decoding
    decoded = Dense(784, activation='sigmoid')(encoded)
    # model
    autoencoder = Model(input_img, decoded)

    # encoder model
    encoder = Model(input_img, encoded)

    # decoder model
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1] # last layer of encoder
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    # optimize model
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    train_model(autoencoder, x_train, x_train, x_test, x_test)

    plot_result(x_test)


# finish
# wow good our model worked wery well
# i am proud of myself
# rauf odilov