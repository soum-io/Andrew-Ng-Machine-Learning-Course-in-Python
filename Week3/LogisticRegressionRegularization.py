'''
By          : Michael Shea
Date        : 5/22/2018
Email       : mjshea3@illinois.edu
Phone       : 708 - 203 - 8272
Description :
This is the code for the Logistic Regression Problem with two categories using
regularization. The Description of the problem can be found in ex2.pdf section
2.
'''

import numpy as np
import scipy as sp
from numpy import linalg as la
from scipy import linalg as sla
import csv
import os
import os.path
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random
import sympy



# Helper function for obtaining prediction of classification model
# input:
#    theta_vals   - current guesses for thetas
#    x_vals       - feature data for prediction
#    print_pred   - if true, function will prind the prediction
# outputs:
#    the prediction result
def Pred_classification(theta_vals, x_vals, print_pred = False):
    # first column of x_vals needs to be ones
    padded_x_vals = np.ones(x_vals.size+1)
    padded_x_vals[1:] = np.copy(x_vals)
    # ensure they are in correcy numpy array shape
    theta_vals.reshape((theta_vals.size,1))
    padded_x_vals.reshape((padded_x_vals.size, 1))
    # transpose of theta times features
    z = theta_vals.T @ padded_x_vals
    # sigmoid function
    score = (1/(1+np.exp(-z)))
    # print score if true
    if(print_pred):
        print("probability that output is 1: " + str(score))
    return score

# Given two initial features, this function creates 25 new features of power
# of at most 6, then calls Pred_classification to predict and print results
# input:
#    theta_vals   - current guesses for thetas
#    x0           - x0 feature for prediction
#    x1           - x1 feature for prediciton
# outputs:
#    None
def pred_xs(x0, x1, theta_vals):
    # data_main will hold all the feature info. there are 27 combinations
    # that can be made of a power of at most 6
    data_main = np.ones((1,27))
    data_main[0,0] = x0
    data_main[0,1] = x1

    # arrays to hold the polyinomial values of each feature to power 6
    feature0_poly = np.zeros((1,6))
    feature1_poly = np.zeros((1,6))
    # get column of initial features
    feature0_poly[:,0] = np.copy(x0)
    feature1_poly[:,0] = np.copy(x1)
    # power up to 6
    for power in range(2,7):
        feature0_poly[:,power-1] = np.power(feature0_poly[:,0],power)
        feature1_poly[:,power-1] = np.power(feature1_poly[:,0],power)


    # var to keep track of current index into main_data, starting at 2 bc first
    # two columns have original feature values in them already
    data_ind = 2
    # input poly data into main_data for values where the features are by
    # themselves (e.i. x0^2, x1^4 and not x0^2*x1^4)
    for single_power in range(1,6):
        data_main[:, data_ind] = feature0_poly[:,single_power]
        data_ind = data_ind + 1

        data_main[:, data_ind] = feature1_poly[:,single_power]
        data_ind = data_ind + 1

    # not input the cross features (like x0^2*x1^4)
    for feature0 in range(feature0_poly.shape[1]):
        for feature1 in range(feature1_poly.shape[1]):
            # make sure it is within 6th power
            if((feature0+1) + (feature1+1) <= 6):
                data_main[:,data_ind] = np.multiply(feature0_poly[:,feature0],
                    feature1_poly[:,feature1])
                data_ind = data_ind + 1
    # print inputs and call input features into sigmoid functionprint
    print("x0: " + str(x0))
    print("x1: " + str(x1))
    Pred_classification(theta_vals, data_main, print_pred = True)


# Helper function for obtaining cost and theta gradients for particular guess.
# Not very usefule by itself
# input:
#    data         - Feature and output data in 2d array format
#    theta_vals   - current guesses for thetas
#    m            - number of samples
#    num_features - the number of features in the dataset
#    lam          - the lambda value for the regularization portion
# outputs:
#    tuple of (in this order) J (cost), gradients of each theta value that has
#    already been mukpltied by alpha
def Cost_and_gradients(data, theta_vals,  m, num_features, lam = 1):
    # inital gradients and cost from inital guess based from cost function
    J = 0
    J_reg = 0
    # one more theta than features
    theta_gradients = np.zeros(num_features+1)

    J = 0
    for row in range(m):
        # calculate the cost step given the current theta parameters
        x_vals = data[row,:num_features]
        prediciton = Pred_classification(theta_vals, x_vals)
        y_i = data[row,num_features]
        J_step = -y_i*np.log(prediciton)-(1-y_i)*np.log(1-prediciton)
        J = J + J_step

        # calulate portion of  gradient
        # first do first gradient
        grad_step = (prediciton-y_i)*1
        theta_gradients[0] = theta_gradients[0] + grad_step
        # now the rest
        for feature in range(num_features):
            grad_step = (prediciton-y_i)*data[row,feature]
            theta_gradients[feature+1] = theta_gradients[feature+1] + grad_step

    # update the regularization portion to cost function and gradients besides
    # theta0
    gradient_reg = np.zeros(num_features+1)
    for theta in range(1, num_features+1):
        J_reg = J_reg + theta_vals[theta]**2
        gradient_reg[theta] = (lam/m)*theta_vals[theta]

    # finishing calualtion for cost
    J = J/m + (lam/(2*m))*J_reg
    # finishing calulation for graidents
    theta_gradients = (1/m)*theta_gradients + gradient_reg
    return J, theta_gradients


# function for performing logistic regression with two variables using regularized
# gradient descent
# input:
#    rel_file_path         - path the csv data file
#    a                     - inital alpha value
#    n                     - max number of iterations that the algorithm should perform
#                            before increasing its alpha value and trying again
# outputs:
#    tuple of (in this order) optimal theta values in an array
def Two_category_classification_reg(rel_file_path, a = 8, n = 2000, lam = 1):
    # list to hold values of cost per iteration
    J_values = []

    # all data as a 2d array
    data = np.loadtxt(rel_file_path, delimiter=',')

    # m is the number of datapoints
    m= data.shape[0]
    # 27 features for feature polynomials of degree 6 discluding features of all
    # ones
    num_features = 27

    # data_main will be used to fill in in the extra features. 27 features in
    # total, plus output values so 28 columns in total
    data_main = np.ones((m,num_features+1))
    data_main[:, :2] = np.copy(data[:,:2])
    data_main[:, 27] = np.copy(data[:,2])

    # arrays to hold the polyinomial values of each feature to power 6
    feature0_poly = np.zeros((m,6))
    feature1_poly = np.zeros((m,6))

    # this 2d array will hold the powers of each individual feature
    powers = np.zeros((2,num_features+1))
    powers[:,1] = np.array([1,0])
    powers[:,2] = np.array([0,1])

    # keep tack of powers indexing
    powers_ind = 3


    # get column of initial features
    feature0_poly[:,0] = np.copy(data[:,0])
    feature1_poly[:,0] = np.copy(data[:,1])
    # power up to 6
    for power in range(2,7):
        feature0_poly[:,power-1] = np.power(feature0_poly[:,0],power)
        feature1_poly[:,power-1] = np.power(feature1_poly[:,0],power)


    # var to keep track of current index into main_data, starting at 2 bc first
    # two columns have original feature values in them already
    data_ind = 2
    # input poly data into main_data for values where the features are by
    # themselves (e.i. x0^2, x1^4 and not x0^2*x1^4)
    for single_power in range(1,6):
        data_main[:, data_ind] = feature0_poly[:,single_power]
        data_ind = data_ind + 1
        powers[:,powers_ind] = np.array([single_power+1,0])
        powers_ind = powers_ind + 1

        data_main[:, data_ind] = feature1_poly[:,single_power]
        data_ind = data_ind + 1
        powers[:,powers_ind] = np.array([0,single_power+1])
        powers_ind = powers_ind + 1

    # not input the cross features (like x0^2*x1^4)
    for feature0 in range(feature0_poly.shape[1]):
        for feature1 in range(feature1_poly.shape[1]):
            # make sure it is within 6th power
            if((feature0+1) + (feature1+1) <= 6):
                data_main[:,data_ind] = np.multiply(feature0_poly[:,feature0],
                    feature1_poly[:,feature1])
                data_ind = data_ind + 1

                powers[:,powers_ind] = np.array([feature0+1,feature1+1])
                powers_ind = powers_ind + 1


    # theta gradients and cost initilization. We will use 1 to start.
    # there is one more theta than features, hence the +1
    theta_gradients = np.full(num_features+1, 1)
    J = np.Inf
    # iterate until optimal is found, n interations has happened, or J is
    # increasing. If n iterations happens, update alpha by multiplying it by
    # 3 to speed up process and start over. If J is increasing, alpha is too
    # large, so divide it by 3 and start over
    # optimal is found when both theta gradients are sufficiently small
    #initialize number of iterations and theta values to 0
    iterations = 0
    theta_vals = np.zeros(num_features + 1)
    while True:
        #bool to hold if we will break or not
        break_out = True
        for gradient in theta_gradients:
            if np.abs(gradient) > 10e-4:
                break_out = False
        if break_out:
            break
        iterations = iterations + 1
        # keep track of previous J to ensure it is not increasing
        J_prev = J
        J, theta_gradients = Cost_and_gradients(data_main, theta_vals,  m,
            num_features, lam)
        # check that iterations is not to high, otherwise the convergence is
        # to slow! If so, increase alpha and reset
        if(iterations > n):
            #J increased, decrease alpha, reset, and start over
            a = a*(1+random.uniform(0, 1))
            print("Trying a bigger alpha value, alpha = " + str(a), end = '\n')
            # set gradients to 1 as a temperary value
            theta_gradients = np.full(num_features, 1)
            #reset thetas and interations
            theta_vals = np.zeros(num_features + 1)
            #reset J_values
            J_values = []
            iterations = 0
            J = np.inf
            continue
        # ensure cost is not increasing
        if(J_prev < J ):
            #increase alpha by factor of 3
            a = a/(1+ random.uniform(0, 1))
            print("Trying a smaller alpha value, alpha = " + str(a), end = '\n')
            # set gradients to 1 as a temperary value
            theta_gradients = np.full(num_features, 1)
            #reset thetas and interations
            theta_vals = np.zeros(num_features + 1)
            #reset J_values
            J_values = []
            iterations = 0
            J = np.inf
            continue
        # update thetas and J_values and scale gradients by alpha
        theta_vals = theta_vals - a*theta_gradients
        J_values.append(J)
    # print the results
    print("The optimal theta paremeters are: " + str(repr(theta_vals)))
    # show plots
    Plot_data_2_cats(data_main, theta_vals, J_values, m, powers, lam)
    return theta_vals


# function for plotting raw data, best-fit line for classification, and Cost
# per iteration plots for classification problem with two categories
# input:
#   data          - feature and output data in 2d array
#   theta_vals    - optimal theta values for classification
#   m             - numer of row, or data points
#   feature_avg   - average of each feature column
#   feature_range - range of each feature column
# outputs:
#    no return value, will display plot figure as subplots
def Plot_data_2_cats(data, theta_vals, J_values, m, powers, lam):
    # create symbolic equation of regularized classification
    x0 = sympy.symbols('x0')
    x1 = sympy.symbols('x1')
    f = 0
    for ele in range(powers.shape[1]):
        f = f + theta_vals[ele] * x0**powers[0,ele] * x1**powers[1,ele]

    # plot original raw data with best fit line
    plt.subplot(221)
    for row in range(m):
        if(data[row,27] == 0):
            plt.scatter([data[row,0]], [data[row,1]], marker = 'o', color = "yellow")
        else:
            plt.scatter([data[row,0]], [data[row,1]], marker = '+', color= "blue")
    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")
    # legend example was taken from this post:
    # https://stackoverflow.com/questions/47391702/matplotlib-making-a-colored-markers-legend-from-scratch
    y0 = mlines.Line2D([], [], color='yellow', marker='o', linestyle='None',
                          markersize=10, label='y = 0')
    y1 = mlines.Line2D([], [], color='blue', marker='+', linestyle='None',
                              markersize=10, label='y = 1')
    x = np.linspace(-1,1.5,1000)
    y = np.linspace(-1,1.5,1000)
    X,Y = np.meshgrid(x,y)
    # use lambda function to convert sympy to matplotlib capable
    F = sympy.lambdify((x0, x1), f, modules=['numpy'])
    # show contour for only when z = 0, this is same as setting the forumla
    # that was obtained using sympy to 0. This is done in the [0] parementer of
    # plt.contour
    plt.contour(X,Y,F(X,Y), [0])
    plt.title("Raw Data With Best-Fit Classifier with Lambda = " + str(lam))
    plt.legend(handles=[y0, y1], loc='upper right')
    plt.grid(True)


    # plot Cost Fucntion to number of iterations
    plt.subplot(222)
    plt.scatter(np.arange(1,len(J_values)+1), np.array(J_values), marker = 'x', color = "red")
    plt.xlabel("# of iterations")
    plt.ylabel("Cost")
    plt.title("Cost per Iteration")
    plt.grid(True)

    plt.show()


def main():
    # solve 2 category classification and print results - try changing lambda
    # to get the results in section 2.5
    data1_path = "machine-learning-ex2/ex2/ex2data2.txt"
    theta_vals = Two_category_classification_reg(data1_path, lam=1, n = 5000)

    # predict probility that output is 1 with x0 being .1 and x1 being .2
    pred_xs(.1 , .2, theta_vals)

# call main
main()
