import numpy as np

class MiniBatchSGD:
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def create_mini_batches(self, X, y, batch_size):
        indices = np.random.permutation(len(y))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        mini_batches = [(X_shuffled[i:i + batch_size], y_shuffled[i:i + batch_size]) for i in range(0, len(y), batch_size)]
        return mini_batches

    def update_weights(self, X, y, w, lambda_):
        step_size = self.learning_rate / (1 + self.learning_rate * lambda_)
        loss = 0
        for i in range(len(y)):
            margin = y[i] * np.dot(X[i], w)
            if margin < 1:
                w = (1 - step_size * lambda_) * w + step_size * y[i] * X[i]
                loss += 1 - margin
            else:
                w = (1 - step_size * lambda_) * w
        reg_loss = 0.5 * lambda_ * np.sum(w ** 2)
        total_loss = (loss / len(y)) + reg_loss
        return w, total_loss

class Adagrad(MiniBatchSGD):

    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gradient_accumulator = None

    def update_weights(self, X, y, w, lambda_):
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
        self.gradient_accumulator += gradient ** 2

        adjusted_learning_rates = self.learning_rate / (np.sqrt(self.gradient_accumulator) + self.epsilon)
        w -= adjusted_learning_rates * gradient

        reg_loss = 0.5 * lambda_ * np.sum(w ** 2)
        total_loss = (loss / len(y)) + reg_loss

        return w, total_loss