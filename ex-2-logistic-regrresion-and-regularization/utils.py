import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd

# ~ #################### ~ #
# utils - helper functions  #
# ~ #################### ~ #

def sigmoid(z):
    """
    compute sigmoid in a numerically stable way to avoid overflow.
    for z >= 0: sigmoid = 1 / (1 + exp(-z))
    for z <  0: sigmoid = exp(z) / (1 + exp(z))
    """
    z = np.array(z, copy=False)
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    neg = ~positive
    # positive branch
    out[positive] = 1 / (1 + np.exp(-z[positive]))
    # negative branch
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1 + exp_z)
    return out

def compute_cost(X, y, theta):
    """
    compute cost and gradient for logistic regression with log clipping.
    avoids log(0) by clipping hypothesis to [eps, 1-eps].
    """
    eps = 1e-15
    y = y.reshape(-1, 1)
    m = y.size

    z = X.dot(theta)
    h = sigmoid(z)
    # clip to avoid log(0)
    h = np.clip(h, eps, 1 - eps)

    J = (-1 / m) * (y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h)))
    grad = (1 / m) * X.T.dot(h - y)
    return J, grad


def costFunctionReg(theta, X, y, lambda_):
    """
    compute regularized cost and gradient for logistic regression,
    with log clipping to prevent divide-by-zero.
    """
    eps = 1e-15
    m = y.shape[0]
    z = X.dot(theta)
    h = sigmoid(z)
    h = np.clip(h, eps, 1 - eps)

    term1 = -y.T.dot(np.log(h))
    term2 = -(1 - y).T.dot(np.log(1 - h))
    reg_cost = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    J = (1 / m) * (term1 + term2) + reg_cost

    error = h - y
    grad = (1 / m) * X.T.dot(error)
    grad[1:] += (lambda_ / m) * theta[1:]
    return J.squeeze(), grad


def gradientDescentReg(X, y, theta, alpha, num_iters, lambda_):
    """
    run gradient descent with regularization.
    returns optimized theta and cost history.
    """
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        J, grad = costFunctionReg(theta, X, y, lambda_)
        theta = theta - alpha * grad
        J_history[i] = J
    return theta, J_history


def gradient_descent(X, y, theta, alpha, num_iters, plot_flag=False): 
    """
    perform vanilla gradient descent to learn theta.
    """
    J_iter = np.zeros((num_iters, 1))
    for i in range(num_iters):
        cost, grad = compute_cost(X, y, theta)
        J_iter[i] = cost[0][0]
        theta = theta - alpha * grad
    if plot_flag:
        plt.plot(J_iter)
        plt.show()
    return theta, J_iter


def plot_logreg_line(X, y, theta):  # ~ professor-provided
    """
    plot data points and logistic regression boundary for 2D input.
    X must include bias term in column 0.
    """
    x1 = X[:,1]
    x2 = X[:,2]
    x1_min, x1_max = 1.1 * x1.min(), 1.1 * x1.max()
    y1_min = -(theta[0] + theta[1] * x1_min) / theta[2]
    y1_max = -(theta[0] + theta[1] * x1_max) / theta[2]

    plt.plot(
        x1[y[:,0]==0], x2[y[:,0]==0], 'ro',
        x1[y[:,0]==1], x2[y[:,0]==1], 'go',
        [x1_min, x1_max], [y1_min, y1_max], 'b-'
    )
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('data and decision boundary')
    plt.grid()
    plt.show()
    

def map_feature(x1, x2, degree=6):  # ~ professor-provided
    """
    map two input features to polynomial features up to given degree.
    returns design matrix with bias column.
    """
    x1 = x1.reshape(-1,1)
    x2 = x2.reshape(-1,1)
    X = np.ones((x1.shape[0], 1))
    for i in range(1, degree+1):
        for j in range(i+1):
            X = np.hstack([X, (x1**(i-j)) * (x2**j)])
    return X


def normalize_feature(x):
    """
    normalize 1-D array to zero mean and unit variance.
    returns (x_norm, mu, sigma).
    """
    mu = np.mean(x)
    sigma = np.std(x)
    return (x - mu) / sigma, mu, sigma


def predict_proba(X, theta):
    """
    return probability vector h = sigmoid(X.dot(theta)).
    """
    return sigmoid(X.dot(theta))


def predict_label(X, theta):
    """
    return binary labels {0,1} by thresholding at 0.5.
    """
    return (predict_proba(X, theta) >= 0.5).astype(int)


def polyFeatureVector(x1, x2, degree):  # ~ professor-provided
    """
    vectorized polynomial feature mapping for two features.
    returns matrix of shape (m, num_poly_terms).
    """
    x1 = x1.reshape(-1,1)
    x2 = x2.reshape(-1,1)
    Xp = np.ones((x1.shape[0], 1))
    for i in range(1, degree+1):
        for j in range(i+1):
            Xp = np.hstack([Xp, x1**(i-j) * x2**j])
    return Xp


def plotDecisionBoundary1(theta, X, y, d):  # ~ professor-provided
    """
    plot decision boundary for polynomial features of degree d.
    """
    x1 = X[:,1]
    x2 = X[:,2]
    plt.plot(
        x1[y[:,0]==0], x2[y[:,0]==0], 'ro',
        x1[y[:,0]==1], x2[y[:,0]==1], 'go'
    )
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('data')
    plt.grid()

    u = np.linspace(x1.min(), x1.max(), 50)
    v = np.linspace(x2.min(), x2.max(), 50)
    z = np.zeros((u.size, v.size))
    for i in range(u.size):
        for j in range(v.size):
            vec = polyFeatureVector(np.array([u[i]]), np.array([v[j]]), d)
            z[i,j] = vec.dot(theta)
    z = z.T
    plt.contour(u, v, z, levels=[0], linewidths=2)
    plt.show()