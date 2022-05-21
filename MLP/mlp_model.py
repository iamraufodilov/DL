# load libraries
import numpy as np 
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

text= ['I like this code',
       'This code seems to me amazing',
       'Who writed this astonishing code',
       'This code written as profeccional, without bugs',
       'This is a best code in human history',
       'What a awful code',
       'Where did you get this messy code',
       'Why this code is so simple and without knowledge',
       'I do not like this kind of simple code',
       'This is really bad code I ever seen'
    ]

target = [1,1,1,1,1,0,0,0,0,0]

X_train, X_test, y_train, y_test = train_test_split(text, target, test_size=0.2, random_state=7)
print(X_train, y_train) # good keep going

vectorizer = TfidfVectorizer(stop_words = 'english', lowercase=True, norm='l1')

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = Perceptron()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = np.round(metrics.accuracy_score(y_pred, y_test), 2)

print("here we go there is accuracy score: {}".format(score))

# actually model did bad because the data is very small and model is very simple.
# rauf odilov 