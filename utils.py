import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


def get_data(dataset):
    """
    Load and preprocess the specified dataset.

    Parameters:
    -----------
    dataset : str
        Name of the dataset ('tiny', 'large', or 'imdb').

    Returns:
    -----------
    tuple or tuple of numpy.ndarrays
        Depending on the dataset:
        - For 'tiny' and 'large' datasets, returns X and y.
        - For 'imdb' dataset, returns X_train, X_test, y_train, and y_test.
    """
    if dataset == "tiny":
        data = pd.read_csv("data/toydata_tiny.csv")
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        return X, y

    if dataset == "large":
        data = pd.read_csv("data/toydata_large.csv")
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        return X, y

    if dataset == "imdb":
        data = np.load("data/imdb.npz", allow_pickle=True)

        X_train = data["train"]
        y_train = data["train_labels"]
        X_test = data["test"]
        y_test = data["test_labels"]

        X_train = X_train.item().todense()
        X_test = X_test.item().todense()

        return X_train, X_test, y_train, y_test


def grid_search_svm(
    X,
    y,
    model,
    optimizer,
    lambdas,
    learning_rates,
    n_epochs=30,
    batch_size=10,
    n_splits=5,
):
    """
    Perform grid search for the best hyperparameters of SVM.

    Parameters:
    -----------
    X : numpy.ndarray
        Features.
    y : numpy.ndarray
        Target labels.
    model : object
        SVM model object.
    optimizer : object
        Optimizer object.
    lambdas : list
        List of regularization parameters.
    learning_rates : list
        List of learning rates.
    epochs : int, optional (default=30)
        Number of epochs for training.
    batch_size : int, optional (default=10)
        Size of mini-batch for training.
    n_splits : int, optional (default=5)
        Number of folds for cross-validation.

    Returns:
    -----------
    tuple
        Best hyperparameters, best accuracy, and dictionary of results.
    """
    kf = KFold(n_splits=n_splits)
    results = {}

    for lr in learning_rates:
        for lambda_ in lambdas:
            accuracies = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model.lambda_ = lambda_
                optimizer.learning_rate = lr
                model.fit(X_train, y_train, optimizer, n_epochs, batch_size)

                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)

            results[(lr, lambda_)] = np.mean(accuracies)

    best_params = max(results, key=results.get)
    return best_params, results[best_params], results


def cross_val_score(X, y, model, optimizer, epochs=30, batch_size=10, n_splits=5):
    """
    Perform cross-validation and calculate average accuracy.

    Parameters:
    -----------
    X : numpy.ndarray
        Features.
    y : numpy.ndarray
        Target labels.
    model : object
        SVM model object.
    optimizer : object
        Optimizer object.
    epochs : int, optional (default=30)
        Number of epochs for training.
    batch_size : int, optional (default=10)
        Size of mini-batch for training.
    n_splits : int, optional (default=5)
        Number of folds for cross-validation.

    Returns:
    -----------
    float
        Average accuracy.
    list
        List of loss histories for each fold.
    """
    kf = KFold(n_splits=n_splits)
    accuracies = []
    all_loss_histories = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        loss_history = model.fit(X_train, y_train, optimizer, epochs, batch_size)
        all_loss_histories.append(loss_history)

        y_pred = model.predict(X_test)
        fold_accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(fold_accuracy)

    average_accuracy = np.mean(accuracies)
    return average_accuracy, all_loss_histories
