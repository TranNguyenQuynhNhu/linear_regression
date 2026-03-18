import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """
        Train the linear regression model using Gradient Descent
        """
        n_samples, n_features = X.shape

        # initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # training loop
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # compute loss (MSE)
            loss = (1 / n_samples) * np.sum((y_pred - y) ** 2)
            self.loss_history.append(loss)

    def predict(self, X):
        """
        Predict using the trained model
        """
        return np.dot(X, self.weights) + self.bias

    def get_params(self):
        """
        Return learned parameters
        """
        return self.weights, self.bias
