'''
By          : Michael Shea
Date        : 6/8/2018
Email       : mjshea3@illinois.edu
Phone       : 708 - 203 - 8272
Description :
This is the code for Recommender System problems. The Description of the problem
can be found in ex8.pdf part 2.
'''

import numpy as np
import scipy as sp
from numpy import linalg as la
from scipy import linalg as sla
import scipy.io
import csv
import os
import os.path
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import random


# function for calcualting the cost of a reccommender system dataset
# input:
#   Y              - num_movies x num_users array of user ratings
#   R              - num_movies x num_users array of 1 if if a user rated a specific movie
#                    and 0 otherwise
#   X              - num_movies x num_features dataset of movie weights
#   theta          - num_users x num_features dataset of user weights
#   lam (optional) - regularization value
# outputs:
#    The cost for the given weight
def Cost(Y, R, X, theta, lam = 0):
    # calculate regularized cost - vectorized form
    cost = .5 * np.sum(np.square(np.multiply(((theta@(X.T)).T - Y), R))) + lam/2 * (np.sum(
        np.square(theta)) + np.sum(np.square(X)))
    return cost

# function for calcualting the gradients of a reccommender system dataset
# input:
#   Y              - num_movies x num_users array of user ratings
#   R              - num_movies x num_users array of 1 if if a user rated a specific movie
#                    and 0 otherwise
#   X              - num_movies x num_features dataset of movie weights
#   theta          - num_users x num_features dataset of user weights
#   lam (optional) - regularization value
# outputs:
#    Two arrays for the gradients of the user weights and movie weights
def Gradient(Y, R, X, theta, lam = 0):
    # regularized gradients - vectorized
    # compute vecotirzed movie gradients
    X_grad = np.zeros(X.shape)
    for movie in range(R.shape[0]):
        # list of all users that have rated current movie
        idx = np.where(R[movie,:] == 1)[0]
        X_grad[movie,:] = np.multiply(np.tile((theta @ (X[movie,:].T) -
            Y[movie,:]),(theta.shape[1],1)).T,theta)[idx, :].sum(axis = 0)

    # compute vecotirzed user graidnets
    theta_grad = np.zeros(theta.shape)
    for user in range(R.shape[1]):
        # list of all movies that current user has rated
        idx = np.where(R[:,user] == 1)[0]
        theta_grad[user,:] = np.multiply(np.tile((theta[user,:] @ (X.T) -
            Y[:,user]),(theta.shape[1],1)).T,X)[idx, :].sum(axis = 0)

    # regularize gradeints
    theta_grad  = theta_grad +   lam * theta
    X_grad = X_grad +  lam * X
    return X_grad, theta_grad


# function for verifying the gradients calulated in Gradient function using
# numerical gradient approxiamtions
# input:
#   Y              - num_movies x num_users array of user ratings
#   R              - num_movies x num_users array of 1 if if a user rated a specific movie
#                    and 0 otherwise
#   X              - num_movies x num_features dataset of movie weights
#   theta          - num_users x num_features dataset of user weights
#   e (optional)   - step value in numerical approximation
#   lam (optional) - regularization value
# outputs:
#    the average relative differnece between the numerical approxiamtion and the
#    calualted gradients. This is not a return value, it is simply printed to
#    the screen.
def Check_gradients(Y,R,X,theta,e = 1e-4, lam = 0):
    X_grad, theta_grad = Gradient(Y, R, X, theta, lam = lam)
    theta_grad_check = np.zeros(theta_grad.shape)
    X_grad_check = np.zeros(X_grad.shape)
    num_movies, num_users = R.shape
    num_features = X.shape[1]

    # test X gradients
    for row in range(num_movies):
        for col in range(num_features):
            X_minus = np.copy(X)
            X_minus[row,col] = X_minus[row,col] - e
            X_plus = np.copy(X)
            X_plus[row,col] = X_plus[row,col] + e
            cost_minus = Cost(Y, R, X_minus, theta, lam = lam)
            cost_plus = Cost(Y, R, X_plus, theta, lam = lam)
            X_grad_check[row,col] = (cost_plus - cost_minus)/(2*e)

    # test theta gradients
    for row in range(num_users):
        for col in range(num_features):
            theta_minus = np.copy(theta)
            theta_minus[row,col] = theta_minus[row,col] - e
            theta_plus = np.copy(theta)
            theta_plus[row,col] = theta_plus[row,col] + e
            cost_minus = Cost(Y, R, X, theta_minus, lam = lam)
            cost_plus = Cost(Y, R, X, theta_plus, lam = lam)
            theta_grad_check[row,col] = (cost_plus - cost_minus)/(2*e)

    # ensure no elements have 0 so we dont get a divide by 0 error
    X_grad = X_grad + 1e-20
    theta_grad = theta_grad + 1e-20

    X_rel_dif = np.sum(X_grad - X_grad_check)/X_grad.size
    print("Relative difference for X is " + str(X_rel_dif))
    theta_rel_dif = np.sum(theta_grad - theta_grad_check)/theta_grad.size
    print("Relative difference for theta is " + str(theta_rel_dif))


def main():
     mat_file_loc = "machine-learning-ex8/ex8/ex8_movies.mat"
     Y = scipy.io.loadmat(mat_file_loc)['Y']
     R = scipy.io.loadmat(mat_file_loc)['R']

     # pre load weights for testing
     mat_file_loc = "machine-learning-ex8/ex8/ex8_movieParams.mat"
     X = scipy.io.loadmat(mat_file_loc)['X']
     theta  = scipy.io.loadmat(mat_file_loc)['Theta']

     # test unregualrized cost (lam = 0) - section 2.2.1
     # reduce data set size to run faster - this is dont in the matlab example
     # as well if you loo kat the code
     num_users, num_movies, num_features = 4, 5, 3
     # will output the expexted 22.22
     print("Unregularized cost: " + str(Cost(Y[:num_movies, :num_users],
        R[:num_movies, :num_users], X[:num_movies,:num_features],
        theta[:num_users, :num_features], lam = 0)))

     # test regualrized cost (lam = 1.5) - section 2.2.3
     # reduce data set size to run faster - this is dont in the matlab example
     # as well if you loo kat the code
     num_users, num_movies, num_features = 4, 5, 3
     # will output the expexted 31.34
     print("Reegularized cost: " + str(Cost(Y[:num_movies, :num_users],
        R[:num_movies, :num_users], X[:num_movies,:num_features],
        theta[:num_users, :num_features], lam = 1.5)))

     # test gradient descent with small data sizes for speed for regularized and
     # unregularized. The relative differnces are around e-12, which means the
     # gradient funciton is working. Sec 2.2.2 and 2.2.4
     print("\nDoing non-regularized gradient checking:")
     Check_gradients(Y[:num_movies, :num_users],  R[:num_movies, :num_users],
        X[:num_movies,:num_features], theta[:num_users, :num_features], lam = 0)
     print("\nDoing regularized gradient checking:")
     Check_gradients(Y[:num_movies, :num_users],  R[:num_movies, :num_users],
        X[:num_movies,:num_features], theta[:num_users, :num_features], lam = 1.5)


    # I have not had the change to actually do a regression model using the cost
    # and gradient functions yet - I plan to do s0 - but it was not required for
    # this section




# call main
main()
