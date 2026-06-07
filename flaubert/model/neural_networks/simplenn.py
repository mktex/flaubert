# Nützliche Imports
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

from sklearn.datasets import make_regression

from flaubert.vis import xdiagramme as xdg


def MSEStep(X, y, W, b, learn_rate=0.005):
    y_pred = W @ X.transpose() + b
    error = y - y_pred
    xfaktor = 1.0 / len(error)
    partial_error_w = -xfaktor * np.mean(error * X)
    partial_error_b = -xfaktor * np.mean(error)
    W_new = W - learn_rate * partial_error_w
    b_new = b - learn_rate * partial_error_b
    return W_new, b_new


def miniBatchGD(X, y, batch_size=20, learn_rate=0.005, num_iter=25, plot_result=True):
    n_points = X.shape[0]
    W = np.random.rand(X.shape[1])
    b = np.random.rand() * y.mean()
    regression_coef = [np.hstack((W, b))]
    for _ in range(num_iter):
        batch = np.random.choice(range(n_points), batch_size)
        X_batch = X[batch, :]
        y_batch = y[batch]
        W, b = MSEStep(X_batch, y_batch, W, b, learn_rate)
        regression_coef.append(np.hstack((W, b)))
    if plot_result:
        X_min = X.min()
        X_max = X.max()
        counter = len(regression_coef)
        for W, b in regression_coef:
            counter -= 1
            color = [1 - 0.999 ** counter for _ in range(3)]
            if counter == 0:
                color = "orange"
            plt.plot([X_min, X_max], [X_min * W + b, X_max * W + b], color=color)
        plt.scatter(X, y, zorder=3, s=7, color="blue")
        plt.title("Annäherung mit SGD Batch-Verfahren")
        plt.tight_layout()
        plt.show()
    return regression_coef


X, y = make_regression(n_samples=750, n_features=1, noise=35, random_state=42)
X = (X - X.min()) / (X.max() - X.min())
y = (y - y.min()) / (y.max() - y.min())
plt.close()
miniBatchGD(X, y, batch_size=50, learn_rate=1.5, num_iter=5000, plot_result=True)
