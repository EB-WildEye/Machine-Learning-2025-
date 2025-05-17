import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd

# ~ #################### ~ #
# Utils - Helper functions #
# ~ #################### ~ #


# ~ professor-provided functions ~ #
def sigmoid(z):
    """ Compute sigmoid function """
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, theta):
    """ Compute cost for logistic regression """
    y = y.reshape(-1, 1)  # Ensure y is a column vector
    m = y.size
    
    grad_J = np.zeros(theta.shape)
    z = np.dot(X, theta)
    h_theta = sigmoid(z)
    
    J = (- 1 / m) * (np.dot(y.T, np.log(h_theta)) + np.dot((1 - y).T, np.log(1 - h_theta)))
    grad_J = 1 / m * np.dot(X.T, (h_theta - y))
    return J, grad_J

def cost_logreg(x, y, theta):
    m = y.size
    J = 0
    grad = np.zeros(theta.shape)
    z = np.dot(x, theta)
    h_theta = sigmoid(z)
    
    epsilon = 1e-5
    h_theta = np.clip(h_theta, epsilon, 1 - epsilon)
    
    J = (-1 / m) * (np.dot(y.T, np.log(h_theta)) + np.dot((1-y).T, np.log(1-h_theta)))
    grad = (1 / m) * np.dot(x.T, (h_theta - y))
    return J, grad


def gradient_descent(X, y, theta, alpha, num_iters, plot_flag=False, log_cost=False):
    """ Perform gradient descent to learn theta """
    J_iter = np.zeros((num_iters, 1))
    for iter in range(num_iters):
        if log_cost:
            cost, grad = cost_logreg(X, y, theta)
        cost, grad = compute_cost(X, y, theta)
        J_iter[iter] = cost[0][0]
        theta = theta - alpha * grad
        
    if plot_flag:
        plt.plot(J_iter), plt.show()
    return theta, J_iter


def plot_logreg_line(X, y, theta):
    """
    plot_reg_line plots the data points and regression line for logistic regrssion
    Input arguments: X - np array (m, n) - independent variable.
    y - np array (m,1) - target variable
    theta - parameters
    The function is for 2-d input - x2 = -(theta[0] + theta[1]*x1)/theta[2]
    """
    ind = 1
    x1_min = 1.1*X[:,ind].min()
    x1_max = 1.1*X[:,ind].max()
    x2_min = -(theta[0] + theta[1]*x1_min)/theta[2]
    x2_max = -(theta[0] + theta[1]*x1_max)/theta[2]
    x1 = X
    x1lh = np.array([x1_min, x1_max])
    x2lh = np.array([x2_min, x2_max])

    x1 = X[: , 1]
    x2 = X[: , 2]
    ### please add here a line to plot the points and the decision boundary
    plt.plot(x1[y[:,0] == 0], x2[y[:,0] == 0], 'ro', x1[y[:,0] == 1], x2[y[:,0] == 1],
            'go', x1lh, x2lh, 'b-')
    plt.xlabel('x1'), plt.ylabel('x2')
    plt.title('data')
    plt.grid(), plt.show()

# \ end of professor part
 
 
# ~ warppars for predictions ~ #
def predict_proba(X, theta):
    """ return probability vector h = sigmoid(X*theta) """
    return sigmoid(X.dot(theta))


def predict_label(X, theta):
    """tresh hold at 0.5 --> {0,1}"""
    return (predict_proba(X, theta) >= 0.5).astype(int)


def normalize_feature(x):
    """
    normalize a 1-D array x to zero mean and unit variance.
    returns: x_norm, mu, sigma
    """
    mu    = np.mean(x)
    sigma = np.std(x)
    return (x - mu) / sigma, mu, sigma


 
 