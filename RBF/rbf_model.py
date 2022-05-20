# load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rbf(x, c, s):
    return np.exp(-1/(2*s**2)*(x-c)**2)

# define kmeans model
def kmeans(x, k):
    clusters = np.random.choice(np.squeeze(x), size = k)
    prevClusters = clusters.copy()
    stds = np.zeros(k)
    
    converged = False

    while not converged:
        distance = np.squeeze(np.abs(x[:, np.newaxis] - clusters[np.newaxis, :]))
        closestCluster = np.argmin(distance, axis=1)

        for i in range(k):
            pointsForCluster = x[closestCluster == i]

            if len(pointsForCluster) > 0:
                clusters[i] = np.mean(pointsForCluster, axis=0)

        converged = np.linalg.norm(clusters - prevClusters) < 1e-6
        prevClusters = clusters.copy()

    distance = np.squeeze(np.abs(x[:, np.newaxis] - clusters[np.newaxis, :]))
    closestCluster = np.argmin(distance, axis = 1)

    clustersWithNoPoints = []
    for i in range(k):
        pointsForCluster = x[closestCluster ==i]
        if len(pointsForCluster)<2:
            clustersWithNoPoints.append(i)
            continue
        else:
            stds[i] = np.std(x[closestCluster == i])

    if len(clustersWithNoPoints) >0:
        pointsToAverage = []
        for i in range(k):
            if i not in clustersWithNoPoints:
                pointsToAverage.append(x[closestCluster == i])

        pointsToAverage = np.concatenate(pointsToAverage).ravel()
        stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))

    return clusters, stds

# create rbf class
class RBFnet(object):
    def __init__(self, k=2, lr=0.01, epochs=50, rbf=rbf, inferStds=True):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds

        self.w = np.random.randn(k)
        self.b = np.random.randn(1)


    def fit(self, X, y):
        if self.inferStds:
            self.centers, self.stds = kmeans(X, self.k)

        else:
            self.centers, _ = kmens(X, self.k)
            dmax = max([np.abs(c1-c2) for c1 in self.centers for c2 in self.centers])
            self.stds = np.repeat(dMax/np.sqrt(2*self.k), self.k)

        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                a = np.array([self.rbf(X[i], c, s) for c, s in zip(self.centers, self.stds)])
                F = a.T.dot(self.w) + self.b

                # forward pass
                loss = (y[i] - F).flatten()**2
                print("Loss: {0:.2f}".format(loss[0]))

                # backward pass
                error = -(y[i] - F).flatten()

                # update w and b
                self.w = self.w - self.lr*a*error
                self.b = self.b - self.lr*error


    # predict function
    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s in zip(self.centers, self.stds)])
            F = a.T.dot(self.w)+self.b
            y_pred.append(F)

        return np.array(y_pred)



# create sample dataset
SAMPLE_INPUTS = 100
X = np.random.uniform(0., 1., SAMPLE_INPUTS)
X = np.sort(X, axis=0)
noise = np.random.uniform(0., 1., SAMPLE_INPUTS)
y = np.sin(2*np.pi*X)+noise

# assign the model
model = RBFnet()

# train the model
model.fit(X, y)

# lets evaluate the model
y_pred = model.predict(X)

plt.plot(X, y, '-o', label='True')
plt.plot(X, y_pred, '-o', label='Predicted')
plt.legend()
plt.show()

# that is awesome after so many unsuccessful projects
# good keep going train 
# rauf odilov