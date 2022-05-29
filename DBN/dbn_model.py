# load necessary libraries
from dbn.tensorflow import SupervisedDBNClassification
import numpy as np
import pandas as pd
from sklearn.metrics.classification import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# load dataset
dataset = pd.read_csv("G:/rauf/STEPBYSTEP/Data/mnist/train.csv")
scaler = StandardScaler()
X = np.array(dataset.drop(["label"], axis=1))
y = np.array(dataset["label"])
X = scaler.fit_transform(X)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1)


# define the model
clasifier = SupervisedDBNClassification(hidden_layers_structure =[256, 256],
                                        learning_rate_rbm=0.05,
                                        learning_rate=0.1,
                                        n_epochs_rbm=10,
                                        n_iter_backprop=100,
                                        batch_size=32,
                                        activation_function='relu',
                                        dropout_p=0.2)
clasifier.fit(x_train, y_train)

y_pred = clasifier.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)
print("Here is our accuracy {}%".format(int(100*accuracy)))

# rauf odilov