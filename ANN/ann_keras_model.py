# load necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.losses import MeanSquaredLogarithmicError


# load dataste
data = pd.read_csv("G:/rauf/STEPBYSTEP/Tutorial/Repeat Knowledge/DL/Artificial Neural Network/Mall_Customers.csv")
print(data.head(5))
# from daata you can see we do not need customer id to make prediction
# so we delete them
# our goal from this data set to predict spending amount
# as you realize our task is regresssion problem

data.drop("CustomerID", axis = 1, inplace = True)
print(data.head(5))

# next thing we have to do is convert dummy variables
# as model need cnumerical data not categorical

dummy = pd.get_dummies(data.Genre)
merged_data = pd.concat([dummy, data], axis=1)
merged_data.drop("Genre", axis=1, inplace = True)
print(merged_data.head(5))
print(len(merged_data))

# next task is to divide data to feature and label
X = merged_data.iloc[:, 0:4].values
y = merged_data.iloc[:, -1]
print(X[:5])
print(y[:5])
# very good lets move to the next step


# split dataset to train and test set
# as we know our data has 200 training examples 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# but here is the last step for data preprocessing
# this step is feature scaling 
# why we need feature scalling is that because our data holds 
# variety range of data and it will affect model performance
# feature scaling will help model to train perfectly
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# next step is to make model
ann_model = tf.keras.models.Sequential()
ann_model.add(tf.keras.layers.Dense(units=6, activation='relu')) # first hidden layer with 6 nodes
ann_model.add(tf.keras.layers.Dense(units = 6, activation = 'relu')) # second hidden layer with 6 nodes
ann_model.add(tf.keras.layers.Dense(units = 1, activation = 'linear')) # output layer with one node

# optimize model
msle = MeanSquaredLogarithmicError()
ann_model.compile(optimizer = 'adam', loss = msle)

# train the model
ann_model.fit(X_train, y_train, batch_size = 10, epochs = 100)

# lets compare predicted output of X_test and y_test
y_predicted = ann_model.predict(X_test)
for i, j in zip(y_test, y_predicted):
    print("Here is actual Spending sore: {} and here is predicted Spending Score: {} ".format(i, j))

# I do not know model is working terrible bad