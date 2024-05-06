import numpy as np


class MiniBatchSGD:
    """
    Stochastic Gradient Descent (SGD) with mini-batch optimization.

    Attributes:
    -----------
    learning_rate : float, optional (default=0.01)
        Learning rate for gradient descent optimization.

    Methods:
    -----------
    create_mini_batches(X, y, batch_size):
        Create mini-batches for SGD training.

    update_weights(X, y, w, lambda_):
        Update weights using SGD with L2 regularization.
    """

    def __init__(self, learning_rate=0.01):
        """
        Initialize MiniBatchSGD object.

        Parameters:
        -----------
        learning_rate : float, optional (default=0.01)
            Learning rate for gradient descent optimization.
        """
        self.learning_rate = learning_rate

    def create_mini_batches(self, X, y, batch_size):
        """
        Create mini-batches for SGD training.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data, where n_samples is the number of samples and n_features is the number of features.
        y : numpy.ndarray
            Target values (class labels).
        batch_size : int
            Number of samples per mini-batch.

        Returns:
        -----------
        list
            List of mini-batches containing tuples of (X_mini, y_mini).
        """
        indices = np.random.permutation(len(y))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        mini_batches = [
            (X_shuffled[i : i + batch_size], y_shuffled[i : i + batch_size])
            for i in range(0, len(y), batch_size)
        ]

        return mini_batches

    def update_weights(self, X, y, w, lambda_):
        """
        Update weights using SGD with L2 regularization.

        Parameters:
        -----------
        X : numpy.ndarray
            Mini-batch of training data.
        y : numpy.ndarray
            Mini-batch of target values (class labels).
        w : numpy.ndarray
            Current weight vector.
        lambda_ : float
            Regularization parameter.

        Returns:
        -----------
        tuple
            Updated weight vector and total loss.
        """
        step_size = self.learning_rate / (1 + self.learning_rate * lambda_)
        loss = 0

        for i in range(len(y)):
            margin = y[i] * np.dot(X[i], w)
            if margin < 1:
                w = (1 - step_size * lambda_) * w + step_size * y[i] * X[i]
                loss += 1 - margin
            else:
                w = (1 - step_size * lambda_) * w

        reg_loss = 0.5 * lambda_ * np.sum(w**2)
        total_loss = (loss / len(y)) + reg_loss

        return w, total_loss


class Adagrad(MiniBatchSGD):
    """
    Adaptive Gradient Descent (Adagrad) optimizer for mini-batch SGD.

    Attributes:
    -----------
    learning_rate : float, optional (default=0.01)
        Learning rate for gradient descent optimization.
    epsilon : float, optional (default=1e-8)
        Small constant for numerical stability.
    gradient_accumulator : numpy.ndarray
        Accumulated squared gradients.

    Methods:
    -----------
    update_weights(X, y, w, lambda_):
        Update weights using Adagrad with L2 regularization.
    """

    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        """
        Initialize Adagrad object.

        Parameters:
        -----------
        learning_rate : float, optional (default=0.01)
            Learning rate for gradient descent optimization.
        epsilon : float, optional (default=1e-8)
            Small constant for numerical stability.
        """
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gradient_accumulator = None

    def update_weights(self, X, y, w, lambda_):
        """
        Update weights using Adagrad with L2 regularization.

        Parameters:
        -----------
        X : numpy.ndarray
            Mini-batch of training data.
        y : numpy.ndarray
            Mini-batch of target values (class labels).
        w : numpy.ndarray
            Current weight vector.
        lambda_ : float
            Regularization parameter.

        Returns:
        -----------
        tuple
            Updated weight vector and total loss.
        """
        if self.gradient_accumulator is None:
            self.gradient_accumulator = np.zeros_like(w)

        gradient = np.zeros_like(w)
        loss = 0

        for i in range(len(y)):
            margin = y[i] * np.dot(X[i], w)
            if margin < 1:
                gradient -= y[i] * X[i]
                loss += 1 - margin

        gradient += lambda_ * w
        self.gradient_accumulator += gradient**2

        adjusted_learning_rates = self.learning_rate / (
            np.sqrt(self.gradient_accumulator) + self.epsilon
        )
        w -= adjusted_learning_rates * gradient

        reg_loss = 0.5 * lambda_ * np.sum(w**2)
        total_loss = (loss / len(y)) + reg_loss

        return w, total_loss
