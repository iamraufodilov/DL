# load required libraries
import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

np.random.seed(1)


# load dataset
data = pd.read_csv("G:/rauf/STEPBYSTEP/Tutorial/Repeat Knowledge/DL/LSTM/airline-passengers.csv", usecols=[1], engine='python')
print(data.head(5)) # here we go we load our data successfully 

# it is time to prepare it to model
def preprocessing(data):
    dataset = data.values
    dataset = dataset.astype('float32')

    # we have to scale the data
    scaler = MinMaxScaler(feature_range = (0,1))
    dataset = scaler.fit_transform(dataset)

    # split data
    train_size = int(len(dataset)*0.7)
    train = dataset[:train_size, :]
    test = dataset[train_size:, :]
    print("The whole data length is: {}, and from that for train is: {}, and for test is : {}".format(len(dataset), len(train), len(test)))

    return train, test

train, test = preprocessing(data)


# create feature and labels from dataset
# why because our data is one collumn and for time series prediction input is first row and 
# feature is second row respectfully for whole data set
def create_data(dataset, look_back = 1):
    X, y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[0:(i+look_back), 0]
        X.append(a)
        y.append(dataset[i+look_back, 0])

    return np.array(X), np.array(y)

X_train, y_train = create_data(train)
X_test, y_test = create_data(test)

print(X_train[:5], y_train[:5])


# LSTM expects data in the shape of [sample, step, feature]
# but now our data is in shape of [sample, feature]
# we have to reshape input data
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


# create model
model = Sequential()
model.add(LSTM(4, input_shape = (1, look_back)))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(X_train, y_train, epochs = 100, batch_size = 1, verbose = 1)


# evaluate the model
y_predicted = model.predict(y_test)
scalar = MinMaxScaler()
reversed_data_y = scalar.inverse_transform(y_test)
reversed_data_X = scalar.inverse_transform(y_predicted)

mse = mean_squared_error(reversed_data_y, reversed_data_X)
print("Here our mean squared error value: {}".format(mse))


# in final something went wrong 
# it has to been seen to to get result