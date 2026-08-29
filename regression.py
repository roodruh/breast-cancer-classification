import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def forward(self, X_train):
        z = X_train @ self.w + self.b
        sigmoid = (1+np.exp(-z))**-1
        return sigmoid

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        n = len(y_train)
        self.w = np.zeros(X_train.shape[1])
        self.b = 0

        for i in range(self.n_iters):
            p = self.forward(X_train)
            error = p - y_train
            dw = (X_train.T @ error) / n
            db = np.mean(error)
            self.w -= self.lr * dw
            self.b -= self.lr * db

            if i % 100 == 0:
                c = -np.mean(y_train * np.log(p) + (1 - y_train) * np.log(1 - p))
                print(f"iter {i}, cost {c:.4f}")

    def predict_probability(self, X):
        return self.forward(X)

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)
            