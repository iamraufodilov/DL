#load necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# create functions 
def find_BMU(SOM, x):
    distSq = (np.square(SOM - x)).sum(axis=2)
    return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)

def update_weights(SOM, train_ex, learn_rate, radius_sq, BMU_coord, step=3):
    g, h = BMU_coord

    if radius_sq < 1e-3:
        SOM[g, h, :] += learn_rate *(train_ex - SOM[g, h, :])
        return SOM

    for i in range(max(0, g-step), min(SOM.shape[0], g-step)):
        for j in range(max(0, h-step), min(SOM.shape[1], h-step)):
            dist_sq = np.squre(i-g) + np.square(j-h)
            dist_func = np.exp(-dist_sq/2/radius_sq)
            SOM[i, j, :] += learn_rate*dist_func*(train_ex-SOM[i,j,:])

    return SOM

def train_SOM(SOM, train_data, learn_rate=.1, radius_sq=1, lr_decay=.1, radius_decay=.1, epochs=10):
    learn_rate_0 = learn_rate
    radius_0 = radius_sq

    for epoch in np.arange(0, epochs):
        random.shuffle(train_data)
        for train_example in train_data:
            g, h = find_BMU(SOM, train_example)
            SOM = update_weights(SOM, train_example, learn_rate, radius_sq, (g, h))

            learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
            radius_sq = radius_0 * np.exp(-epoch * radius_decay)

    return SOM


# create dataset
# dimesnsions of SOM grid
m=10
n=10
# number of training examples
n_x = 3000

rand = np.random.RandomState(0)

# initialize training data
train_data = rand.randint(0, 255, (n_x, 3))

# initialize SOM randomly
SOM = rand.randint(0, 255, (m, n, 3)).astype('float')

# display both training matrix and SOM
fig, ax = plt.subplots(
    nrows=1, ncols=2, figsize=(12, 3.5),
    subplot_kw=dict(xticks=[], yticks=[]))
ax[0].imshow(train_data.reshape(50, 60, 3))
ax[0].title.set_text('Training Data')
ax[1].imshow(SOM.astype(int))
ax[1].title.set_text('Randomly initialized SOM grid')
#plt.show()

# lets train our model
fig, ax = plt.subplots(
    nrows=1, ncols=2, figsize=(12, 3.5),
    subplot_kw=dict(xticks=[], yticks=[]))

total_epochs = 0
for epoch, i in zip([1,4,5,10], range(0,4)):
    total_epochs += epoch
    SOM = train_SOM(SOM, train_data, epochs=epoch)
    ax[i].imshow(SOM.astype(int))
    ax[i].title.set_text('Epoch = ' + str(total_epochs))
    
plt.show()