import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd

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


def gradient_descent(X, y, theta, alpha, num_iters, plot_flag=False):
    """ Perform gradient descent to learn theta """
    J_iter = np.zeros((num_iters, 1))
    for iter in range(num_iters):
        cost, grad = compute_cost(X, y, theta)
        print(cost[0][0].shape)
        J_iter[iter] = cost[0][0]
        theta = theta - alpha * grad
        
    if plot_flag:
        plt.plot(J_iter), plt.show()
    return theta, J_iter


def compute_cost_reg(X, y, theta, lambda_):
    """
    compute cost and gradient for logistic regression with l2 regularization.
      X       – (m, n) design matrix with intercept column
      y       – (m, 1) binary labels
      theta   – (n, 1) parameters
      lambda_ – regularization strength (float)
    returns (cost, grad)
    """
    m = y.size
    # unregularized cost and gradient
    cost, grad = compute_cost(X, y, theta)
    # regularization term for cost (skip theta[0])
    reg_cost = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    cost = cost + reg_cost

    # regularize gradient (leave theta[0] unchanged)
    grad[1:] += (lambda_ / m) * theta[1:].reshape(-1,1)
    return cost, grad


def gradient_descent_reg(X, y, theta, alpha, num_iters, lambda_, plot_flag=False):
    """
    perform gradient descent with l2 regularization.
      X, y, theta, alpha, num_iters as before
      lambda_ – regularization strength
    returns (theta_opt, cost_history)
    """
    cost_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        cost, grad = compute_cost_reg(X, y, theta, lambda_)
        cost_history[i] = cost
        theta = theta - alpha * grad

    if plot_flag:
        plt.plot(cost_history)
        plt.xlabel("iteration")
        plt.ylabel("cost")
        plt.title(f"regularized gd (lambda={lambda_})")
        plt.grid()
        plt.show()

    return theta, cost_history

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
 
 