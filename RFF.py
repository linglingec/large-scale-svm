import numpy as np

class RandomFourierFeatures:

    def __init__(self, n_components=100, gamma=1.0):
        self.n_components = n_components
        self.gamma = gamma
        self.weights = None 
        self.bias = None

    def fit(self, X):
        d = X.shape[1]
        self.weights = np.random.normal(0, np.sqrt(2 * self.gamma), size=(d, self.n_components))
        self.bias = np.random.uniform(0, 2 * np.pi, size=self.n_components)

    def transform(self, X):
        projection = np.dot(X, self.weights) + self.bias
        return np.cos(projection) * np.sqrt(2. / self.n_components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)