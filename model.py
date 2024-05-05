import numpy as np
from tqdm import tqdm
import time

class LinearSVM:

    def __init__(self, lambda_=0.01):
        self.lambda_ = lambda_
        self.w = None

    def fit(self, X, y, optimizer, epochs=30, batch_size=10):
        self.w = np.zeros(X.shape[1])
        loss_history = []
        pbar = tqdm(range(epochs), desc='Training Progress')
        
        for epoch in pbar:
            mini_batches = optimizer.create_mini_batches(X, y, batch_size)
            epoch_losses = []

            for X_mini, y_mini in mini_batches:
                self.w, loss = optimizer.update_weights(X_mini, y_mini, self.w, self.lambda_)
                epoch_losses.append(loss)

            loss_history.append(np.mean(epoch_losses))
            pbar.set_postfix({'loss': loss_history[-1]})

        return loss_history

    def predict(self, X):
        return np.sign(X.dot(self.w))