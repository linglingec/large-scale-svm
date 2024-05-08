import numpy as np
from tqdm import tqdm
import time


class LinearSVM:
    """
    Linear Support Vector Machine (SVM) classifier.

    Parameters:
    -----------
    lambda_ : float, optional (default=0.01)
        Regularization parameter.

    Attributes:
    -----------
    lambda_ : float
        Regularization parameter.
    w : numpy.ndarray
        Weight vector.

    Methods:
    -----------
    fit(X, y, optimizer, epochs=30, batch_size=10):
        Train the SVM classifier on the given data.

    predict(X):
        Predict class labels for samples in X.
    """

    def __init__(self, lambda_=0.01):
        """
        Initialize LinearSVM object.

        Parameters:
        -----------
        lambda_ : float, optional (default=0.01)
            Regularization parameter.
        """
        self.lambda_ = lambda_
        self.w = None

    def fit(self, X, y, optimizer, n_epochs=30, batch_size=10):
        """
        Train the SVM classifier on the given data.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data, where n_samples is the number of samples and n_features is the number of features.
        y : numpy.ndarray
            Target values (class labels).
        optimizer : Object
            Optimization algorithm object with methods create_mini_batches and update_weights.
        epochs : int, optional (default=30)
            Number of iterations over the training dataset.
        batch_size : int, optional (default=10)
            Number of samples per mini-batch.

        Returns:
        -----------
        loss_history : list
            List of loss values for each epoch.
        """
        self.w = np.zeros(X.shape[1])
        loss_history = []
        pbar = tqdm(range(n_epochs), desc="Training Progress")

        for epoch in pbar:
            mini_batches = optimizer.create_mini_batches(X, y, batch_size)
            epoch_losses = []

            for X_mini, y_mini in mini_batches:
                self.w, loss = optimizer.update_weights(
                    X_mini, y_mini, self.w, self.lambda_
                )
                epoch_losses.append(loss)

            loss_history.append(np.mean(epoch_losses))
            pbar.set_postfix({"loss": loss_history[-1]})

        return loss_history

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        -----------
        X : numpy.ndarray
            Test samples.

        Returns:
        -----------
        numpy.ndarray
            Predicted class labels.
        """
        return np.sign(X.dot(self.w))
