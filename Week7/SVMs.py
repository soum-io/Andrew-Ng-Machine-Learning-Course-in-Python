'''
By          : Michael Shea
Date        : 5/30/2018
Email       : mjshea3@illinois.edu
Phone       : 708 - 203 - 8272
Description :
This is the code for SVM problems. The Description of the problem can be found
in ex6.pdf part 1.
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


# function for visualling classifier of a dataset that was orrginally two
# variables
# input:
#   clf - a classifier
#   X   - Training data
#   y   - training solution
#   xx  - meshgrid ndarray
#   yy  - meshgrid ndarray
# outputs:
#    countour graph of decision boundary
def plot_contours(clf, X, y, xx, yy):
    # plot contour, partialy used from
    # # http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z)

    # plot data
    # number of data points
    m = X.shape[0]
    # positive examples have '+' marker, while negative example have 'o'
    for row in range(m):
        if(y[row] == 0):
            plt.scatter([X[row,0]], [X[row,1]], marker = 'o', color = "yellow")
        else:
            plt.scatter([X[row,0]], [X[row,1]], marker = '+', color= "black")
    plt.show()

# function for solving model using SVM.
# input:
#   X                 - Training data
#   y                 - training solution
#   C (optional)      - C value in SVM algorithm, used for regularization
#   sig (optional)    - sig value used in gaussian elimination
#   kernel (optional) - type of SVM kernel to use
#   graph (optional)  - If the data orginally has two variables, graph the
#                       classifiers output
# outputs:
#    Weight values (except for gaussian elimination), and classifier
def SVM_solve(X, y, C = 1, sig = 1, kernel = "linear", graph = True):
    # define svm for the classifier. SVC is scikit learn's SVM with C param
    # support
    if(kernel == "gaussian"):
        # 'rbf' kernel is same as gaussian - stands for 'Radial basis function'
        clf = SVC(C = C, kernel = "rbf", gamma = sig)
        # train model on data
        # ensure y is correct shape
        y = y.reshape(y.size)
        clf.fit(X, y)
        # obtain meshgrid arrays to plot contour of nonlinear classifier
        x_axis =  np.linspace(np.amin(X[:,0]),np.amax(X[:,0]),100)
        y_axis =  np.linspace(np.amin(X[:,1]),np.amax(X[:,1]),100)
        xx, yy = np.meshgrid(x_axis, y_axis)
        if(graph):
            plot_contours(clf,X, y, xx, yy)
        # return classifier
        return clf

    else:
        clf = SVC(C = C, kernel = kernel)
        # train model on data
        # ensure y is correct shape
        y = y.reshape(y.size)
        clf.fit(X, y)
        # obtain optimized theta parameters
        theta_vals = np.zeros(X.shape[1]+1)
        theta_vals[0] = clf.intercept_
        theta_vals[1:] = clf.coef_
        if(graph):
            Plot_two_cat_class(X,y,theta_vals)

        # return optimal theta parameters and classifier
        return theta_vals, clf

# function for plotting the datapoints of a two category classification with two
# variables
# input:
#   X                     - Training data
#   y                     - Training solution
#   theta_vals (optional) - plot linear kernel decision boundary by supplying
#                           the weights
# outputs:
#    Plot of category info
def Plot_two_cat_class(X, y, theta_vals = np.array([])):
    # number of data points
    m = X.shape[0]
    # positive examples have '+' marker, while negative example have 'o'
    for row in range(m):
        if(y[row] == 0):
            plt.scatter([X[row,0]], [X[row,1]], marker = 'o', color = "yellow")
        else:
            plt.scatter([X[row,0]], [X[row,1]], marker = '+', color= "black")
    if(theta_vals.size > 0):
        horiz_axis = np.linspace(np.amin(X[:,0]),np.amax(X[:,0]),1000)
        # equation if acting like x0 is x_axis and x1 is vert axis
        vert_axis =  np.array([(-theta_vals[0]-theta_vals[1]*x_ele)/theta_vals[2]
            for x_ele in horiz_axis])
        plt.plot(horiz_axis,vert_axis)
    plt.show()

# function for visualizing the best C and sig value to use for SVM with
# gaussian kernel
# input:
#   C_arr             - array of C values to test with
#   sig_arr           - array of sig values to test with
#   X                 - Training data
#   y                 - training solution
#   Xval              - test data set to test model with
#   yval              - test solution set to test model with
#   graph (optional)  - if two class and features, plot the C ans sig data to
#                       visualize the optimal solution
# outputs:
#   graph of data if desired, and then tuple consiting of the best C and sig value
def Best_C_and_sig(C_arr, sig_arr, X, y, Xval, yval, graph = True):
    # loop through all combinations and find the best accuracy on validation set
    best_C = C_arr[0]
    best_sig = sig_arr[0]
    best_accuracy = 0
    for C in C_arr:
        for sig in sig_arr:
            # obtain classifier
            clf = SVM_solve(X, y, C = C, sig = sig, kernel = "gaussian", graph = False)
            # get accuracy and update if neccessary
            accuracy = clf.score(Xval, yval)
            if(C == C_arr[0] and sig == sig_arr[0]):
                best_accuracy = accuracy
            else:
                if(accuracy > best_accuracy):
                    best_accuracy, best_C, best_sig = accuracy, C, sig
    # print and graph results
    print("Best C value is " + str(best_C) + " and the best sig value is " + str(best_sig))
    if(graph):
        SVM_solve(X, y, C = best_C, sig = best_sig, kernel = "gaussian", graph = True)
    # return results
    return best_C, best_sig


def main():
    # data used in section 1.1
    # To open the .mat files, I used the method found in this post:
    # https://stackoverflow.com/questions/874461/read-mat-files-in-python
    mat_file_loc = "machine-learning-ex6/ex6/ex6data1.mat"
    X = scipy.io.loadmat(mat_file_loc)['X']
    y = scipy.io.loadmat(mat_file_loc)['y']

    # plot data to visualize it (sec 1.1)
    Plot_two_cat_class(X,y)

    # Visualize SVM with C = 1 (sec 1.1)
    SVM_solve(X, y, C = 1, graph = True)

    # Visualize SVM with C = 1 (sec 1.1)
    SVM_solve(X, y, C = 100, graph = True)

    # data used in section 1.2
    # To open the .mat files, I used the method found in this post:
    # https://stackoverflow.com/questions/874461/read-mat-files-in-python
    mat_file_loc = "machine-learning-ex6/ex6/ex6data2.mat"
    X = scipy.io.loadmat(mat_file_loc)['X']
    y = scipy.io.loadmat(mat_file_loc)['y']

    # plot data to visualize it (sec 1.2)
    Plot_two_cat_class(X,y)

    # Visualize SVM with kernel = "gaussian". (sec 1.2)
    SVM_solve(X, y, C = 1000000, sig =10, kernel = "gaussian", graph = True)

    # data used in section 1.2.3
    # To open the .mat files, I used the method found in this post:
    # https://stackoverflow.com/questions/874461/read-mat-files-in-python
    mat_file_loc = "machine-learning-ex6/ex6/ex6data3.mat"
    X = scipy.io.loadmat(mat_file_loc)['X']
    y = scipy.io.loadmat(mat_file_loc)['y']
    Xval = scipy.io.loadmat(mat_file_loc)['Xval']
    yval = scipy.io.loadmat(mat_file_loc)['yval']

    # plot data to visualize it (sec 1.2.3)
    Plot_two_cat_class(X,y)

    # obtain best C and sig values. Note the values are not defined the exact
    # same way in the class as they are in scikit learn's implimentation. Read
    # the documentation for svm.SCV if you are interested in the exact
    # differences
    C_arr = np.array([0.01,0.03,0.1,0.3,1,3,10,30])
    sig_arr = np.array([0.01,0.03,0.1,0.3,1,3,10,30])
    Best_C_and_sig(C_arr, sig_arr, X, y, Xval, yval, graph = True)



# call main
main()
