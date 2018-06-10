'''
By          : Michael Shea
Date        : 5/23/2018
Email       : mjshea3@illinois.edu
Phone       : 708 - 203 - 8272
Description :
This is the code for forward propagation in nueral networks.
The Description of the problem can be found in ex3.pdf section 2.
The neural net has 400 input nodes, 25 hidden layers nodes (1 hidden layer),
and 10 output nodes for each digit.
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
import matplotlib.lines as mlines
import random
import sympy



# Solves sigmoid formula for input z
# input:
#    z  - input to sigmoid formula
# outputs:
#    value of sigmoid formula with input z
def Sigmoid_function(z):
    # sigmoid function
    return 1/(1+np.exp(-z))


# function for predicting the category of a 400x25x10 neural network problem.
# Works with mulpitple or single datapoints, and will calculate the accuracy if
# solutions are supplied
# input:
#    data                     - data file containing features to the classification
#                               problem
#    theta1_data              - data for optimized theta_values to hidden layer.
#                               Shape of KxM, where K is number of hidden nodes
#                               and theta is number of input parameters.
#    theta2_data              - data for optimized theta_values to output layer.
#                               Shape of KxM, where K is number of output categories
#                               and theta is number of hidden layer nodes.
#    print_accuracy(optional) - State whether solutions are provided and the
#                               accuracy should be printed
#    solutions(optional)      - Solution array to data
# outputs:
#    returns an array for all the predictions and prints the accuracy of the
#    predictions if print_accuracy is true and solutions are supplied
def Predict_category_nn(data, theta1_data, theta2_data, print_accuracy = False, solutions = 0):
    # number of data points to predict
    num_data = data.shape[0]
    # number of nodes for hidden layer
    num_nodes = theta1_data.shape[0]
    # number of categories for output layer
    num_cats = theta2_data.shape[0]
    # prediction array
    predictions = np.zeros(num_data)
    # var to hold accuracy info, only usefile when print_accuracy is true
    num_true = 0
    # loop through each data input
    for data_segment in range(num_data):
        # get features data padded with one for bias
        x_vals = np.ones((1, data.shape[1]+1))
        x_vals[0,1:] = np.copy(data[data_segment, :])
        # probability for node in hidden layer
        node_prob = np.zeros(num_nodes)
        # input layer to hidden later calculations
        for node in range(num_nodes):
            # first set of theta values
            theta1_vals = np.copy(theta1_data[node,:]).reshape((theta1_data.shape[1], 1))
            # z for sigmoid function
            z = x_vals @ theta1_vals
            node_prob[node] = Sigmoid_function(z)
        # hidden layer to output layer calculations
        # get node data padded with one for hidden layer bias
        node_vals = np.ones((1, node_prob.size+1))
        node_vals[0,1:] = np.copy(node_prob)
        # probability for each category
        cat_prob = np.zeros(num_cats)
        for cat in range(num_cats):
            # second set of theta values
            theta2_vals = np.copy(theta2_data[cat,:]).reshape((theta2_data.shape[1], 1))
            # z for sigmoid function
            z = node_vals @ theta2_vals
            cat_prob[cat] = Sigmoid_function(z)
        # in octave and matlab, arrays are 1 indexed. So digit one was the First
        # category, and digit 0 was the last (10th category). The next line
        # accounts for this
        prediction = np.argmax(cat_prob) + 1
        predictions[data_segment] = prediction
        # keep track of number of right predictions if required
        if(print_accuracy):
            if(int(predictions[data_segment]) == solutions[data_segment]):
                num_true = num_true + 1

    if(print_accuracy):
        print("Accuracy: " + str(num_true / num_data))
    # return predictions
    return predictions



def main():
    # To open the .mat files, I used the method found in this post:
    # https://stackoverflow.com/questions/874461/read-mat-files-in-python
    digit_data= scipy.io.loadmat("machine-learning-ex3/ex3/ex3data1.mat")['X']
    digit_solutions = scipy.io.loadmat("machine-learning-ex3/ex3/ex3data1.mat")['y']

    # pre calulated theta weights for forward propagation
    theta1_data = scipy.io.loadmat("machine-learning-ex3/ex3/ex3weights.mat")["Theta1"]
    theta2_data = scipy.io.loadmat("machine-learning-ex3/ex3/ex3weights.mat")["Theta2"]

    # get predictions, will get the 97.5% specified in ex3.pdf - sec 2
    predictions = Predict_category_nn(digit_data, theta1_data, theta2_data,
        print_accuracy = True, solutions = digit_solutions)

# call main
main()
