# load libraries
import pandas as pd
import numpy as np
from PIL import Image
import os
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import InputLayer

# image loader function
def load_image(image_path):
    image_array = []
    image_name = []
    for folder in os.listdir(image_path):
        for file in os.listdir(os.path.join(image_path, folder)):
            file_path = os.path.join(image_path, folder, file)
            image = np.array(Image.open(file_path))
            image = np.resize(image, (128, 128, 3))
            image = image.astype('float32')
            image /=255
            image_array.append(image)
            image_name.append(folder)

    return image_array, image_name


# convert categorical label to numeric
def convert_categorical(item):
    my_set = set(img_label)
    numeric_label = []
    for i in img_label:
        num = list(my_set).index(i)
        numeric_label.append(num)
    return np.array(numeric_label)


# create model
def create_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(128, 128, 3)))
    model.add(Conv2D(32,(3,3), activation = 'relu'))
    model.add(MaxPool2D(2,2))
    model.add(Flatten())

    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model


# train the model
def train_model(X_train, y_train, model):
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.fit(X_train, y_train, epochs = 10)

    return model

def prediction(img_array, img_name, trained_model):
    random_data = random.choice(img_array)
    index_random = img_array.index(random_data)
    random_label = img_name[index_random]
    my_set = set(img_name)
    my_list = list(my_set)
    my_dict = {}
    for i in my_list:
        if i not in my_dict:
            my_dict[my_list.index(i)] = i
        else:
            pass

    y_predicted = trained_model.predict(random_data)
    print("Our humble prediction is {}. Is it correct?".format(my_dict.get(y_predicted)))
     


data_path ='G:/rauf/STEPBYSTEP/Data2/Custom/horse vs zebra'


img_feature, img_label = load_image(data_path)
img_label2 = convert_categorical(img_label) # in order to convert categorical data
print(img_feature[:5], img_label2[:5])

model = create_model()

trained_model = train_model(img_feature, img_label2, model)