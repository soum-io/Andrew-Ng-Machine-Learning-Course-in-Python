'''
By          : Michael Shea
Date        : 6/6/2018
Email       : mjshea3@illinois.edu
Phone       : 708 - 203 - 8272
Description :
This is the code for anomaly detection problems. The Description of the problem
can be found in ex8.pdf part 1.
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


# function for finding the mean and standard deviation of of each feature of a
# dataset
# input:
#   X - dataset
# outputs:
#    2xM array, where M is number of features of dataset. First row is means,
#    second row is stndard deviations
def Mean_and_std(X):
    # array that will hold the mean and standard deviation values for each
    # feature. First row is mean, second row is average.
    mean_std = np.zeros((2, X.shape[1]))
    for feature in range(X.shape[1]):
        mean_std[0,feature] = np.average(X[:,feature])
        mean_std[1,feature] = np.std(X[:,feature])
    return mean_std

# function for calualting the gaussian distribution of a single feature
# input:
#   x        - datapoint
#   feature  - feature to perform gaussian prob on
#   mean_std - array returned from Mean_std
# outputs:
#    double, probability value
def Single_prob_dist(x, feature, mean_std):
    # obtain mean and standard deviations of the feature
    mean = mean_std[0,feature]
    std = mean_std[1,feature]
    # compute components of probability dist function
    left_side = 1/(np.sqrt(2*np.pi*std**2))
    right_side = np.exp(-(x-mean)**2/(2*std**2))
    return left_side * right_side

# function for calualting the gaussian distribution of a single datapoint with
# multiple features. Assumes independence of features
#   x        - datapoint with mult features (ndarary)
#   mean_std - array returned from Mean_std
# outputs:
#    double, probability value
def Mult_prob_dist(x, mean_std):
    # init gaussian prob
    prob = 1
    for feature in range(mean_std.shape[1]):
        prob = prob * Single_prob_dist(x[feature],feature, mean_std)
    return prob

# function for plotting countour of gaussian distributions of a 2 feature
# dataset
#   X             - dataset, Nx2 where N is the number of data points
#   ep (optional) - probability cutoff boundary to label anomolies
# outputs:
#    plot of original dataset with probability contour
def Gaus_contour(X, ep = 0):
    mean_std = Mean_and_std(X)
    # prepare contour plot
    x_axis = np.linspace(np.amin(X[:,0]), np.amax(X[:,0]), 100)
    y_axis = np.linspace(np.amin(X[:,1]), np.amax(X[:,1]), 100)
    xx, yy = np.meshgrid(x_axis, y_axis)
    zz = np.zeros(xx.shape)

    for x_idx, x in enumerate(x_axis):
        for y_idx, y in enumerate(y_axis):
            # print(str(x) + " " + str(y) + " " + str( Mult_prob_dist([x,y],mean_std)))
            zz[y_idx,x_idx] = Mult_prob_dist([x,y],mean_std)

    # plot the contour of the distribution
    plt.contourf(xx,yy,zz, 100, cmap = 'RdGy')
    plt.colorbar()
    # plot data points
    plt.scatter(X[:,0], X[:,1], marker = 'x')
    plt.ylabel("Throughput (mb/s)")
    plt.xlabel("Latency (ms)")
    # mark outliers
    if(ep is not 0):
        for x_ele in range(X.shape[0]):
            if(Mult_prob_dist(X[x_ele,:],mean_std) < ep):
                plt.scatter([X[x_ele,0]], [X[x_ele,1]], c = 'y')
    plt.show()


# function for calualting the best ep value of a dataset
#   X    - Training dataset
#   Xval - cross validation dataset
#   yval - cross validation dataset solutions
# outputs:
#    tuple consisting of array of probabilities from the cross validations dataset
#    and also the best ep value to use to determine if a datapoint is an anomaly
def Gaus_probs(X, Xval, yval):
    mean_std = Mean_and_std(X)
    # create array to hold probabilitites
    probs = np.zeros((Xval.shape[0],1))
    for data_ele in range(Xval.shape[0]):
        probs[data_ele,0] = Mult_prob_dist(Xval[data_ele,:],mean_std)
    # keep decreasing ep until optimal f1 score is a
    ep_best = np.NINF
    f1_best = np.NINF
    step_size = (np.amax(probs) - np.amin(probs))/1000
    ep_vals = np.arange(np.amin(probs), np.amax(probs), step_size)
    for ep in ep_vals:
        # keep track of scoring params
        tp, fp, fn = 0 , 0 , 0
        for data_ele in range(Xval.shape[0]):
            if(probs[data_ele,0] > ep and yval[data_ele,0] == 1):
                # false negative
                fn = fn + 1
            elif(probs[data_ele,0] < ep and yval[data_ele,0] == 1):
                # true positive
                tp = tp + 1
            elif(probs[data_ele,0] < ep and yval[data_ele,0] == 0):
                # false positive
                fp = fp + 1
        # calulate f1 score
        # avoid devide by 0
        if(tp + fp == 0 or tp+fn == 0):
            continue
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        f1 = 2*prec*rec/(prec+rec)
        # track best ep value
        if(f1 > f1_best):
            f1_best = f1
            ep_best = ep
            # print(str(tp) + " " + str(fp) + " " + str(fn))
    return probs, ep_best

# function for counting the estimated number of anomalies of a dataset
#   Xval - cross validation dataset
#   X    - dataset
#   ep   - probability cutoff boundary to label anomolies
# outputs:
#    the number of estimated anomalies
def Count_anomalies(Xval,X, ep):
    mean_std = Mean_and_std(X)
    # counter for anomalies
    anom = 0
    for x_ele in range(Xval.shape[0]):
        if(Mult_prob_dist(Xval[x_ele,:],mean_std) < ep):
            anom = anom + 1
    return anom



def main():
     mat_file_loc = "machine-learning-ex8/ex8/ex8data1.mat"
     X = scipy.io.loadmat(mat_file_loc)['X']
     Xval = scipy.io.loadmat(mat_file_loc)['Xval']
     yval = scipy.io.loadmat(mat_file_loc)['yval']

     # plot data - sec 1.1
     plt.scatter(X[:,0], X[:,1], marker = 'x')
     plt.ylabel("Throughput (mb/s)")
     plt.xlabel("Latency (ms)")
     plt.show()

     # plot gaussian contour similiar to one shown in sec 1.2
     Gaus_contour(X)

     # obtain ep, get same result for ep as stated in example - sec 1.3
     probs, ep = Gaus_probs(X, Xval,yval)
     print("Best epsilon value is " + str(ep))
     Gaus_contour(X, ep=ep)

     # multi feature data set with exact results described in example - sec 1.4
     mat_file_loc = "machine-learning-ex8/ex8/ex8data2.mat"
     X = scipy.io.loadmat(mat_file_loc)['X']
     Xval = scipy.io.loadmat(mat_file_loc)['Xval']
     yval = scipy.io.loadmat(mat_file_loc)['yval']
     probs, ep = Gaus_probs(X, Xval,yval)
     num_anom = Count_anomalies(X,X, ep)
     print("Best epsilon value is " + str(ep) + " and the number of anomalies" +
            " detected is "+ str(num_anom))






# call main
main()
