import numpy as np


class RandomFourierFeatures:
    """
    Approximate kernel mapping using Random Fourier Features.

    Parameters:
    -----------
    n_components : int, optional (default=100)
        Number of Fourier components to generate.
    gamma : float, optional (default=1.0)
        Scaling factor for Fourier components.

    Attributes:
    -----------
    n_components : int
        Number of Fourier components to generate.
    gamma : float
        Scaling factor for Fourier components.
    weights : numpy.ndarray
        Weight matrix for random Fourier features.
    bias : numpy.ndarray
        Bias vector for random Fourier features.

    Methods:
    -----------
    fit(X):
        Fit Random Fourier Features to the data.

    transform(X):
        Transform data into the random Fourier feature space.

    fit_transform(X):
        Fit Random Fourier Features to the data and transform it simultaneously.
    """

    def __init__(self, n_components=100, gamma=1.0):
        """
        Initialize RandomFourierFeatures object.

        Parameters:
        -----------
        n_components : int, optional (default=100)
            Number of Fourier components to generate.
        gamma : float, optional (default=1.0)
            Scaling factor for Fourier components.
        """
        self.n_components = n_components
        self.gamma = gamma
        self.weights = None
        self.bias = None

    def fit(self, X):
        """
        Fit Random Fourier Features to the data.

        Parameters:
        -----------
        X : numpy.ndarray
            Input data.

        Returns:
        -----------
        None
        """
        d = X.shape[1]
        self.weights = np.random.normal(
            0, np.sqrt(2 * self.gamma), size=(d, self.n_components)
        )
        self.bias = np.random.uniform(0, 2 * np.pi, size=self.n_components)

    def transform(self, X):
        """
        Transform data into the random Fourier feature space.

        Parameters:
        -----------
        X : numpy.ndarray
            Input data.

        Returns:
        -----------
        numpy.ndarray
            Transformed data.
        """
        projection = np.dot(X, self.weights) + self.bias
        return np.cos(projection) * np.sqrt(2.0 / self.n_components)

    def fit_transform(self, X):
        """
        Fit Random Fourier Features to the data and transform it simultaneously.

        Parameters:
        -----------
        X : numpy.ndarray
            Input data.

        Returns:
        -----------
        numpy.ndarray
            Transformed data.
        """
        self.fit(X)
        return self.transform(X)
