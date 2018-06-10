'''
By          : Michael Shea
Date        : 5/27/2018
Email       : mjshea3@illinois.edu
Phone       : 708 - 203 - 8272
Description :
This is the code for the Regularized Linear Regression Problem. The
Description of the problem can be found in ex5.pdf.
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
from sklearn import linear_model
import random

# function that takes in single variable data, and returns multiple features of
# the single variable in polynomial form (second row is x^2, third is x^3, etc.)
# after the data has been scales already
# input:
#   data              - M single var data points
#   p                 - order of polynomial desired
#   scaled (optional) - boolean dictating if the data should be scaled first
# outputs:
#    MxN array, where M is number of data points, N is order of polynomial.
#    If scaled is true, will also return the average and standard deviation
#    used to scale the data
def Polynomials_of_single_var(data,p, scaled = False):
    # scale data if desired
    if(scaled):
        avg = np.average(data)
        # get standard deviation
        data_range = np.std(data)
        data = (data - avg)/data_range
    # var to hold final data values
    poly_data = np.zeros((data.size, p))
    # compute column powers to obtain the new features
    for power in range(1,p+1):
        poly_data[:,power-1] = np.power(data,power).reshape(data.size)
    # return relivant data
    if(scaled):
        return poly_data,avg, data_range
    return poly_data

# function that takes in single variable data, and returns multiple features of
# the single variable in polynomial form (second row is x^2, third is x^3, etc.)
# after the data has been scales already
# input:
#   data             - M single var data points
#   p                - order of polynomial desired
#   avg_p            - average that data set had subracted from it if it
#                      has been scaled
#   data_range_p     - range of dataset or standard deviation of dataseet
#                      that devided the data if it was scaled
# outputs:
#    MxN array, where M is number of data points, N is order of polynomial
def Polynomials_of_single_var_pre_scaled(data,p,avg_p, data_range_p):
    # scale data
    data = (data - avg_p)/data_range_p
    # var to hold final data values
    poly_data = np.zeros((data.size, p))
    # compute column powers to obtain the new features
    for power in range(1,p+1):
        poly_data[:,power-1] = np.power(data,power).reshape(data.size)
    # return data
    return poly_data

# function for calculting hypothesis of single data points (function name is
# terrible, I know. I plan to go through and fux it)
# input:
#   theta_vals             - current guess of theta paramerters for linear
#                            regression
#   data                   - input training feature data of M features
# outputs:
#    hypothesis for one data points
def Single_reg_cost(theta_vals, data):
    # keep track of the polynomial power
    power = 0
    # insert 1 into data array to account for theta bias term
    data = np.insert(data,0,1)
    # ensure vectors are right Shape
    theta_vals = theta_vals.reshape((1,theta_vals.size))
    data = data.reshape((data.size,1))
    # calculate and return cost
    cost = theta_vals@data
    return cost

# function for calulation of cost of single data points
# input:
#   theta_vals             - current guess of theta paramerters for linear
#                            regression
#   x_axis                 - input training feature data of M features
#   y_axis                 - output training data of size 1
# outputs:
#    cost for one data point
def Cost_val(theta_vals, x_axis, y_axis):
    # number of datapoints
    m = x_axis.shape[0]
    # pad x_axis with ones
    x_axis_padded = np.ones((x_axis.shape[0], x_axis.shape[1]+1))
    x_axis_padded[:,1:] = np.copy(x_axis)
    # ensure correct shapes
    theta_vals = theta_vals.reshape((1,theta_vals.size))
    # calculate array of hypothesis values
    h_vals = np.zeros((m,1))
    for i in range(m):
        h_vals[i] = Single_reg_cost(theta_vals, x_axis[i,:])
    # ensure solution vector is correct Shape
    y_axis = y_axis.reshape((y_axis.size,1))
    # calculate cost
    J = h_vals - y_axis
    J =(1/(2*m))* np.sum(np.square(J))
    return J

# function for printint values vs error for training and cross validation data
# versus number of data points
# input:
#   train_x                 - input training feature data. NxM array where N is
#                             # of data points, and M is the number of features
#   train_y                 - output training data. Size N, where N is is number
#                             of data points
#   cross_x                 - some as cross_x, but for cross_validation data
#   cross_y                 - some as cross_y, but for cross_validation data
#   num_range               - tuple of min and max number of datapoints to use
#   avg_p (optional)        - average that data set had subracted from it if it
#                             has been scaled
#   data_range_p (optional) - range of dataset or standard deviation of dataseet
#                             that devided the data if it was scaled
# outputs:
#    Prints the graph of the the error of the training and cross validation data
#    versus number of data points (horizontal axis)
def Learning_curve_lin_reg(train_x, train_y, cross_x, cross_y, num_range,
    lam = 0, avg_p = 0, data_range_p = 0):
    # vars to hold error information. +1 to account for zero indexing
    train_err = np.zeros(num_range[1]+1)
    cross_val_err = np.zeros(num_range[1]+1)
    for iterations in range(num_range[0], num_range[1]+1):
        theta_vals = Mult_var_gradient_descent(train_x[:iterations,:],
            train_y[:iterations,:], graph = False, avg = avg_p,
            data_range = data_range_p, lam = lam)
        # Obtain err values for this set of thetas
        J_train = Cost_val(theta_vals, train_x[:iterations,:], train_y[:iterations,:])
        J_test = Cost_val(theta_vals, cross_x, cross_y)
        # update error aarrays
        train_err[iterations] = J_train
        cross_val_err[iterations] = J_test
    # Plot data
    horiz_axis = np.arange(num_range[1]+1)
    plt.plot(horiz_axis[num_range[0]:], train_err[num_range[0]:], label = "Train")
    plt.plot(horiz_axis[num_range[0]:], cross_val_err[num_range[0]:], label = "Cross Validation")
    plt.xlabel("Number of training examples")
    plt.ylabel("Error")
    plt.title("Learning curve for linear regression with Lam = " + str(lam) +
    " and " + str(train_x.shape[1]) + " number of features")
    plt.legend(loc='upper left')
    plt.show()

# function for performing linear regression with multiple var using gradient
# descent. Only one var initially, but can have multiple features of one var
# due to polynomial
# input:
#   data                  - input feature data. NxM array where N is number
#                           of data points, and M is the number of features
#   y_vals                - output data. Size N, where N is is number of data
#                           points
#   graph (optional)      - boolean value if the output should be graphed. Only
#                           should be true if single var features
#   avg (optional)        - average that data set had subracted from it if it
#                           has been scaled
#   data_range (optional) - range of dataset or standard deviation of dataseet
#                           that devided the data if it was scaled
#   lam (optional)        - Lambda value to use for regularization
#   n (optional)          - Max number of iteration to use before changing
#                           alpha value
# outputs:
#    Print the graph of the results if told to do so, and returns the optimized
#    theta parameters
def Mult_var_gradient_descent(data, y_vals, graph = True, avg = 0,
    data_range = 0, lam = 0, n = 10000):

    # define Classifier. Set fit_intecept to false because we will have already
    # regularized data. Redge is sklearns linear regression with regularization.
    # The alpha parameter is the regularization coefficient
    clf = linear_model.Ridge(alpha = lam, max_iter = n, solver = "lsqr")
    clf.fit(data,y_vals)
    # m is datapoints and num features is the number of features
    m, num_features = data.shape
    # get theta_vals from trained classifier
    theta_vals = np.zeros((num_features+1))
    theta_vals[0] = clf.intercept_
    theta_vals[1:] = clf.coef_

    # print results
    # this means data was scaled
    if(avg != 0 and data_range != 0):
        print("The Optimal Formula is y = {0:.2f}".format(theta_vals[0]), end = "")
        for feature in range(num_features):
            print("+{0:.2f}*((x-{1:.2f})/{2:.2f})^{3:.2f} ".format(
                theta_vals[feature+1], avg, data_range, feature,),end="")
        print("",end="\n")
    else:
        print("The Optimal Formula is y = {0:.2f}".format(theta_vals[0]), end = "")
        for feature in range(num_features):
            print("+{0:.2f}*x^{1:.0f}".format(theta_vals[feature+1], feature+1),end="")
        print("",end="\n")
    # print plot of regression line if one one varaible
    if(graph):
        plt.scatter(data[:,:1], y_vals, marker = 'x', color = "red")
        plt.xlabel("Change in water level (x)")
        plt.ylabel("Water flowing out of the dam (y)")
        if(avg != 0 and data_range != 0):
            plt.title("Scaled Data with Best-Fit Line with Lam = " + str(lam))
        else:
            plt.title("Raw Data with Best-Fit Line with Lam = " + str(lam))
        # calculate data points for best fit curve - assumes one var with
        # polynomial features
        # change linspace bounds to see differnet rengions of curve
        horiz_axis = np.linspace(np.min(data[:,:1]), np.max(data[:,:1]))
        vert_axis = np.zeros(horiz_axis.shape)
        for index in range(horiz_axis.size):
            temp_adder = 0
            for theta in range(theta_vals.size):
                temp_adder = temp_adder + theta_vals[theta] * (
                    horiz_axis[index] ** theta)
            vert_axis[index] = temp_adder
        plt.plot(horiz_axis, vert_axis)
        plt.show()

    # return results
    return theta_vals

# function for printint values vs error for training and cross validation data
# versus lambda values
# input:
#   train_x                 - input training feature data. NxM array where N is
#                             # of data points, and M is the number of features
#   train_y                 - output training data. Size N, where N is is number
#                             of data points
#   cross_x                 - some as cross_x, but for cross_validation data
#   cross_y                 - some as cross_y, but for cross_validation data
#   lam_values              - sorted array of lambda values to be tested
#   avg_p (optional)        - average that data set had subracted from it if it
#                             has been scaled
#   data_range_p (optional) - range of dataset or standard deviation of dataseet
#                             that devided the data if it was scaled
# outputs:
#    Prints the graph of the the error of the training and cross validation data
#    versus lambda value (horizontal axis)
def Best_lam(train_x, train_y, cross_x, cross_y, lam_values,
    avg_p = 0, data_range_p = 0):
    # array to hold error info
    train_err = np.zeros(lam_values.size)
    cross_val_err = np.zeros(lam_values.size)
    for lam in range(lam_values.size):
        theta_vals = Mult_var_gradient_descent(train_x,
            train_y, graph = False, avg = avg_p,
            data_range = data_range_p, lam = lam_values[lam])
        # Obtain err values for this set of thetas
        J_train = Cost_val(theta_vals, train_x, train_y)
        J_test = Cost_val(theta_vals, cross_x, cross_y)
        # update error aarrays
        train_err[lam] = J_train
        cross_val_err[lam] = J_test
    # Plot data
    plt.plot(lam_values, train_err, label = "Train")
    plt.plot(lam_values, cross_val_err, label = "Cross Validation")
    plt.xlabel("lambda")
    plt.ylabel("Error")
    plt.legend(loc='upper left')
    plt.show()






def main():
    # To open the .mat files, I used the method found in this post:
    # https://stackoverflow.com/questions/874461/read-mat-files-in-python
    mat_file_loc = "machine-learning-ex5/ex5/ex5data1.mat"
    water_level_train= scipy.io.loadmat(mat_file_loc)['X']
    water_flow_train = scipy.io.loadmat(mat_file_loc)['y']
    water_level_test= scipy.io.loadmat(mat_file_loc)['Xval']
    water_flow_test = scipy.io.loadmat(mat_file_loc)['yval']
    water_level_test_final= scipy.io.loadmat(mat_file_loc)['Xtest']
    water_flow_test_final = scipy.io.loadmat(mat_file_loc)['ytest']

    # plot data to visualize it sec 1.1
    total_level_data = np.concatenate((water_level_train, water_level_test))
    total_flow_data = np.concatenate((water_flow_train, water_flow_test))
    plt.scatter(water_level_train, water_flow_train, marker = 'x', color = "red")
    plt.xlabel("Change in water level (x)")
    plt.ylabel("Water flowing out of dam (y)")
    plt.show()

    # Do single example with one var regularized lin regression - sec 1.2-4
    Mult_var_gradient_descent(water_level_train, water_flow_train)

    # Fit learning curve for single var water example to show high bias.
    # Adds more data points and visualizes training and cross validation error.
    # sec 2.1
    Learning_curve_lin_reg(water_level_train, water_flow_train,
        water_level_test, water_flow_test, (2,12))

    # compute polynomial linear regression
    # obtain new data with polynomial features - sec 3.1
    max_polynomial = 8
    water_level_train_p ,avg_p, range_p = Polynomials_of_single_var(
        water_level_train, max_polynomial, scaled = True)
    Mult_var_gradient_descent(water_level_train_p, water_flow_train,
        avg = avg_p, data_range = range_p)

    # Fit learning curve for multiple polynomial features from a  single var for
    # water example to show high bias.
    # Adds more data points and visualizes training and cross validation error.
    # sec 3.2
    water_level_train_p ,avg_p, range_p = Polynomials_of_single_var(
        water_level_train, 8, scaled = True)
    water_level_test_p = Polynomials_of_single_var_pre_scaled(
        water_level_test, 8, avg_p, range_p)
    Learning_curve_lin_reg(water_level_train_p, water_flow_train,
        water_level_test_p, water_flow_test, (2,12), avg_p = avg_p,
        data_range_p = range_p)

    # Try it with lam = 1 instead of 0
    Mult_var_gradient_descent(water_level_train_p, water_flow_train,
        avg = avg_p, data_range = range_p, lam = 1)
    Learning_curve_lin_reg(water_level_train_p, water_flow_train,
        water_level_test_p, water_flow_test, (2,12), avg_p = avg_p,
        data_range_p = range_p, lam = 1)

    # Try it with lam = 100
    Mult_var_gradient_descent(water_level_train_p, water_flow_train,
        avg = avg_p, data_range = range_p, lam = 100)
    Learning_curve_lin_reg(water_level_train_p, water_flow_train,
        water_level_test_p, water_flow_test, (2,12), avg_p = avg_p,
        data_range_p = range_p, lam = 100)

    # computer errors for different lambdas. Note graphs looks differnt bc
    # different libraries are used and the graphs are in terms of the scaled data
    # sec 3.3
    lam_values = np.array([0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10])
    Best_lam(water_level_train_p, water_flow_train, water_level_test_p,
        water_flow_test, lam_values, avg_p = avg_p, data_range_p = range_p)

    # using the best value of lambda = 3, test it on data not seen yet - sec 3.3
    water_level_test_final_p = Polynomials_of_single_var_pre_scaled(
        water_level_test_final, 8, avg_p, range_p)
    Learning_curve_lin_reg(water_level_train_p, water_flow_train,
        water_level_test_final_p, water_flow_test_final, (2,12), avg_p = avg_p,
        data_range_p = range_p, lam = 3)

    # randomly choose 12 data points for Train and cross val and see how model
    # holds up with lam = .01 - sec 3.5
    total_level_data = np.concatenate([water_level_train, water_level_test, water_level_test_final])
    total_flow_data = np.concatenate([water_flow_train, water_flow_test, water_flow_test_final])
    sample_indices = random.sample(range(0, total_level_data.size), 24)
    train_indices = sample_indices[:12]
    cross_val_indices = sample_indices[12:]
    train_data = np.array([total_level_data[i] for i in train_indices])
    train_solutions = np.array([total_flow_data[i] for i in train_indices])
    cross_data = np.array([total_level_data[i] for i in cross_val_indices])
    cross_solutions = np.array([total_flow_data[i] for i in cross_val_indices])
    # make 8 degree ploynomial features
    train_data ,avg_p, range_p = Polynomials_of_single_var(
        train_data, 8, scaled = True)
    cross_data = Polynomials_of_single_var_pre_scaled(
        cross_data, 8, avg_p, range_p)
    Learning_curve_lin_reg(train_data, train_solutions,
        cross_data, cross_solutions, (2,12), avg_p = avg_p,
        data_range_p = range_p, lam = .01)



# call main
main()
