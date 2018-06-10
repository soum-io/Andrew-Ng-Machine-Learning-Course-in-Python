'''
By          : Michael Shea
Date        : 5/21/2018
Email       : mjshea3@illinois.edu
Phone       : 708 - 203 - 8272
Description :
This is the code for the Logistic Regression Problem with two categories using
gradient descent. The Description of the problem can be found in ex2.pdf section
1.
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
# for scikit learns version of logistic regression
from sklearn import linear_model

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
        print(score)
    return score


# Helper function for obtaining cost and theta gradients for particular guess.
# Not very usefule by itself
# input:
#    data         - Feature and output data in 2d array format
#    theta_vals   - current guesses for thetas
#    m            - number of samples
#    num_features - the number of features in the dataset
# outputs:
#    tuple of (in this order) J (cost), gradients of each theta value that has
#    already been mukpltied by alpha
def Cost_and_gradients(data, theta_vals,  m, num_features):
    # inital gradients and cost from inital guess based from cost function
    J = 0
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

    # finishing calualtion for cost
    J = J/m
    # finishing calulation for graidents
    theta_gradients = (1/m)*theta_gradients
    return J, theta_gradients


# function for performing linear regression with one variable using gradient
# descent
# input:
#    rel_file_path         - path the csv data file
#    a                     - inital alpha value
#    n                     - max number of iterations that the algorithm should perform
#                            before increasing its alpha value and trying again
# outputs:
#    tuple of (in this order) optimal theta values in an array
#    the average of the features, and then the range of the features
def Two_category_classification(rel_file_path, a = 8, n = 2000):
    # list to hold values of cost per iteration
    J_values = []

    # all data as a 2d array
    data = np.loadtxt(rel_file_path, delimiter=',')

    # m is the number of datapoints
    m, num_features = data.shape
    # num features is one les than number of columns bc output is discluded
    num_features = num_features - 1

    # arrays to hold the averages and ranges of each feature
    feature_avg = np.zeros(num_features)
    feature_range = np.zeros(num_features)
    for feature in range(num_features):
        feature_avg[feature] = np.average(data[:,feature])
        feature_range[feature] = np.ptp(data[:,feature])
    # scale feature data with info from above
    for feature in range(num_features):
        for row in range(m):
            data[row,feature] = (data[row,feature] -
                feature_avg[feature])/feature_range[feature]


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
            if np.abs(gradient) > 10e-6:
                break_out = False
        if break_out:
            break
        iterations = iterations + 1
        # keep track of previous J to ensure it is not increasing
        J_prev = J
        J, theta_gradients = Cost_and_gradients(data, theta_vals,  m, num_features)
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
    # show plots if feasible
    if(num_features == 2):
        Plot_data_2_cats(data, theta_vals, J_values, m, feature_avg, feature_range)
    return theta_vals, feature_avg, feature_range


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
def Plot_data_2_cats(data, theta_vals, J_values, m,  feature_avg, feature_range):
    # method to store data this way was discovered from this post:
    # https://stackoverflow.com/questions/44603609/python-how-to-plot-classification-data

    # plot original raw data with best fit line
    plt.subplot(221)
    for row in range(m):
        if(data[row,2] == 0):
            plt.scatter([data[row,0]], [data[row,1]], marker = 'o', color = "yellow")
        else:
            plt.scatter([data[row,0]], [data[row,1]], marker = '+', color= "blue")
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    # legend example was taken from this post:
    # https://stackoverflow.com/questions/47391702/matplotlib-making-a-colored-markers-legend-from-scratch
    admitted = mlines.Line2D([], [], color='yellow', marker='o', linestyle='None',
                          markersize=10, label='Admitted')
    not_admitted = mlines.Line2D([], [], color='blue', marker='+', linestyle='None',
                              markersize=10, label='Not Admitted')
    plt.legend(handles=[admitted, not_admitted])
    x = np.linspace(-.5,.5,1000)
    y = np.array([(-theta_vals[0]-theta_vals[1]*x_ele)/theta_vals[2] for x_ele in x])
    plt.plot(x,y)
    plt.title("Scaled Raw Data With Best-Fit Classifier")
    plt.grid(True)


    # plot Cost Fucntion to number of iterations
    plt.subplot(222)
    plt.scatter(np.arange(1,len(J_values)+1), np.array(J_values), marker = 'x', color = "red")
    plt.xlabel("# of iterations")
    plt.ylabel("Cost")
    plt.title("Cost per Iteration")
    plt.grid(True)

    plt.show()


# Results of using scikit learn's logistic regression.
# input:
#   rel_file_path - path to data
# outputs:
#   No return value. Plots scatter of raw data with best fit line from
#   scikit learn logistic regression model
def Using_Scikit_learn(rel_file_path):
    # get data
    data = np.loadtxt(rel_file_path, delimiter=',')
    X = data[:,:2]
    Y = data[:,2]

    # model info can be found here
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.fit
    clf = linear_model.LogisticRegression(penalty = "l1")
    clf.fit(X,Y)
    # m is datapoints and num features is the number of features
    m, num_features = data.shape
    num_features = num_features-1

    # plot raw data in scatter plot
    for row in range(m):
        if(data[row,2] == 0):
            plt.scatter([data[row,0]], [data[row,1]], marker = 'o', color = "yellow")
        else:
            plt.scatter([data[row,0]], [data[row,1]], marker = '+', color= "blue")
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    # legend example was taken from this post:
    # https://stackoverflow.com/questions/47391702/matplotlib-making-a-colored-markers-legend-from-scratch
    admitted = mlines.Line2D([], [], color='yellow', marker='o', linestyle='None',
                          markersize=10, label='Admitted')
    not_admitted = mlines.Line2D([], [], color='blue', marker='+', linestyle='None',
                              markersize=10, label='Not Admitted')
    plt.legend(handles=[admitted, not_admitted])
    x = np.linspace(30,100,1000)
    # plot best fit with parameters from classifier
    y = np.array([(-clf.intercept_-clf.coef_[0,0]*x_ele)/clf.coef_[0,1] for x_ele in x])
    plt.plot(x,y)
    plt.title("Scikit Learn Logistic Regression")
    plt.show()


def main():
    # solve 2 category classification and print results - will plot data and
    # best fit classifier - sec 1.1 and 1.2
    data1_path = "machine-learning-ex2/ex2/ex2data1.txt"
    theta_vals, feature_avg, feature_range = Two_category_classification(data1_path)

    #predict student with exam 1 score of 45 and exam 2 score of 85
    #scaled exam 1 score
    exam1 = (45 - feature_avg[0])/feature_range[0]
    #scaled exam 2 score
    exam2 = (85 - feature_avg[1])/feature_range[1]
    # print info and results. Results will show .776 as described in sec 1.2.4
    print("Scaled exam scores with original exam 1 of 45% and exam 2 of 85%: "
        + str(exam1)  + ", " + str(exam2))
    x_vals = np.array([exam1, exam2])
    print("Probability student will be admitted:")
    Pred_classification(theta_vals, x_vals, print_pred = True)

    # see what scikit learn's model would look like - to verify results 
    Using_Scikit_learn(data1_path)

# call main
main()
