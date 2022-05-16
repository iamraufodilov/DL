# in this model I will make the class to convert images into csv file
# to make the model to read them vithout difficulty

# load libraries
import numpy as np
from PIL import Image
import matplotlib.image as img
import os
import pandas as pd

'''
# load image
image = Image.open("G:/rauf/STEPBYSTEP/Data2/Custom/horse vs zebra/horse/2Q__ (1).jpg")
image_to_array = np.asarray(image)
print(image_to_array.shape)


# so till now we loaded image 
# the next task is to convert it to 2d dimension as csv accept only 2d dataset
# however our data is 3d because of rgb
image_2D = image_to_array.reshape(image_to_array.shape[0], -1)
print(image_2D.shape)
np.savetxt("G:/rauf/STEPBYSTEP/Tutorial/Repeat Knowledge/DL/Convolutional NN/single_image_converted.csv", image_2D)
'''


def image_loader(folder):
    image_array = []
    for img in os.listdir(folder):
        img_path = folder + "/" + img
        image = Image.open(img_path)
        image_resized = image.resize((28,28))
        img_to_array = np.asarray(image_resized)
        print(img_to_array.shape)
        image_array.append(img_to_array)

    return image_array

def img_2D_to_3D(image_array):
    image_array_list = []
    for img in image_array:
        image_2D = img.reshape(img.shape[0], -1)
        print(image_2D.shape)
        image_array_list.append(image_2D)
    return image_array_list


def save_csv(data):
    np.savetxt("G:/rauf/STEPBYSTEP/Data2/Custom/horse vs zebra/horse.csv", data)



folder = "G:/rauf/STEPBYSTEP/Data2/Custom/horse vs zebra/horse" 

image_array = image_loader(folder)
image_array_2D = img_2D_to_3D(image_array)

save_csv(image_array_2D)