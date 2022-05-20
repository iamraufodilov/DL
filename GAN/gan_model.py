# load libraries
from numpy import zeros, ones, expand_dims, asarray
from numpy.random import randn, randint
from keras.datasets import fashion_mnist
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import Conv2D, Conv2DTranspose, Concatenate
from keras.layers import LeakyReLU, Dropout, Embedding
from keras.layers import BatchNormalization, Activation
from keras import initializers
from keras.initializers import RandomNormal
from keras.optimizers import Adam, RMSprop, SGD
from matplotlib import pyplot
import numpy as np
from math import sqrt

# load data
(X_train, _), (_, _) = fashion_mnist.load_data()
X_train = X_train.astype('float32') / 127.5 -1
X_train = np.expand_dims(X_train, axis = 3)
print(X_train.shape)


# generate latent points
def generate_latent_points(latent_dim, n_samples):
    x_input = np.randn(latent_dim * n_samples)
    z_inputs = x_input.reshape(n_samples, latent_dim)
    return z_inputs

def generate_real_sampels(X_train, n_sampels):
    ix = np.randint(0, X_train.shape[0], n_sampels)
    X = X_train[ix]
    y = np.ones((n_sampels, 1))
    
    return X, y

def generate_face_sampels(generator, latent_dim, n_sampels):
    z_inputs = generate_latent_points(latent_dim, n_sampels)
    image = generator.predict(z_inputs)
    y = np.zeros((n_sampels, 1))

    return image, y

def summarize_performance(step, g_model, latent_dim, n_samples=100):
    X, _ = generate_face_sampels(g_model, latent_dim, n_sampels)
    X = (X+1)/2.0
    for i in range(n_samples):
        plt.subplot(10, 10, 1+i)
        plt.axis('off')
        plt.imshow(X[i, :, :, 0], cmap = 'gray_r')
    filename = 'model_gan.h5' % (step+1)
    g_model.save(filename)
    print("Model saved")

def save_plot(sampels, n_exampels):
    for i in range(n_sampels):
        plt.subplot(sqrt(sampels), sqrt(n_exampels), 1+i)
        plt.axis('off')
        plt.imshow(exampels[i, :, :, 0], cmap = 'gray_r')

    plt.show()


# build the model
def define_discriminator(in_shape=(28, 28, 1)):
    def define_discriminator(in_shape=(28, 28, 1)):
        init = RandomNormal(stddev=0.02)  
        in_image = Input(shape=in_shape)
        fe = Flatten()(in_image)
        fe = Dense(1024)(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Dropout(0.3)(fe)
        fe = Dense(512)(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Dropout(0.3)(fe)
        fe = Dense(256)(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Dropout(0.3)(fe)
        out = Dense(1, activation='sigmoid')(fe)
        model = Model(in_image, out)
        opt = Adam(lr=0.0002, beta_1=0.5) 
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

discriminator = define_discriminator()

def define_generator(latent_dim):
    init = RandomNormal(stddev=0.02)
    in_lat = Input(shape=(latent_dim,)) 
    gen = Dense(256, kernel_initializer=init)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dense(512, kernel_initializer=init)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dense(1024, kernel_initializer=init)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dense(28 * 28 * 1, kernel_initializer=init)(gen)
    out_layer = Activation('tanh')(gen)
    out_layer = Reshape((28, 28, 1))(gen)
    model = Model(in_lat, out_layer)
    return model

generator = define_generator()

def define_gan(g_model, d_model):
    d_model.trainable = False
    gan_output = d_model(g_model.output)
    model = Model(g_model.input, gan_output)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

