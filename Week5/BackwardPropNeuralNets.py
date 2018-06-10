'''
By          : Michael Shea
Date        : 5/24/2018
Email       : mjshea3@illinois.edu
Phone       : 708 - 203 - 8272
Description :
This is the code for backward propagation in nueral networks.
The Description of the problem can be found in ex4.pdf section 1.
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
from PIL import Image


# function showing stitched image of 100 random handwritten digits
# input:
#   digit_data - 2d array of grayscale values of all digits (must be flattened)
# outputs:
#    10 x 10 stiched images of random hand written digits
def Print_100_digits(digit_data):
    # array of 100 random unique indices that are in the range of digit samples
    rand_indices = np.array(random.sample(range(0,digit_data.shape[0]), 100))

    # final image that will have 100 digits stitched together
    final_image = np.zeros((20*10,20*10))
    # keep track of where in final_image we are storing into
    img_count = 0
    img_row = 0
    img_col = 0
    for index in rand_indices:
        # the digits are from 20x20 pixel images. Have to rotate 270 degrees then
        # flip along vertical axis to get desired orientation. Multpiply by 255 so
        # digits can be seen using PIL's image library
        digit = np.flip(np.rot90(np.copy(digit_data[index,:]).reshape((20,20))*255, 3), 1)
        final_image[img_row:img_row+20, img_col:img_col+20] = np.copy(digit)
        # update final image positioning
        img_count = img_count + 1
        img_col = img_col + 20
        if(img_count%10 == 0):
            img_row = img_row + 20
            img_col = 0
    # create and show image
    digits_image = Image.fromarray(final_image)
    digits_image.show()

# Solves sigmoid formula for input z
# input:
#    z  - input to sigmoid formula
# outputs:
#    value of sigmoid formula with input z
def Sigmoid_function(z):
    # sigmoid function
    return 1/(1+np.exp(-z))

# Solves the gradient of the sigmoid formula for input z
# input:
#    z  - input to sigmoid formula
# outputs:
#    value of the gradient of the sigmoid formula with input z
def Sigmoid_gradient(z):
    # note that this does not work for vectors. Numpy's vectorize is used for this.
    # Computer Derivative of sigmoid function
    sigmoid_function_val = Sigmoid_function(z)
    return sigmoid_function_val*(1-sigmoid_function_val)

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
#    predictions if print_accuracy is true and solutions are supplied, also
#    retuns all the output data for each prediction probability for each datapoint
#    for all the categories and hidden nodes. Also returns the z data that goes
#    into the sigomoid formula in the hidden layer.
def Predict_category_nn(data, theta1_data, theta2_data, print_accuracy = False, solutions = 0):
    # number of data points to predict
    num_data = data.shape[0]
    # number of nodes for hidden layer
    num_nodes = theta1_data.shape[0]
    # number of categories for output layer
    num_cats = theta2_data.shape[0]
    # prediction array
    predictions = np.zeros(num_data)
    # sigmoid function data used in nnCost_and_gradient_function
    h_data = np.zeros((num_data, num_cats))
    # sigmoid function data for hidden layer used in nnCost_and_gradient_function
    hidden_layer_data = np.zeros((num_data, num_nodes))
    # z data to go into sigmoid function on second layer - used in
    # nnCost_and_gradient_function
    z_2_data = np.zeros((num_data, num_nodes))
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
            z_2_data[data_segment,node] = z
            node_prob[node] = Sigmoid_function(z)
        hidden_layer_data[data_segment,:] = np.copy(node_prob)
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
        h_data[data_segment,:] = np.copy(cat_prob)
        predictions[data_segment] = prediction
        # keep track of number of right predictions if required
        if(print_accuracy):
            if(int(predictions[data_segment]) == solutions[data_segment]):
                num_true = num_true + 1

    if(print_accuracy):
        print("Accuracy: " + str(num_true / num_data))
    # return predictions and signoid function data for hidden and output nodes
    return predictions, h_data, hidden_layer_data, z_2_data

# Solves the cost cost function of a neural net for a single iteration.
# input:
#    sigmoit_data - array of outputs from the sigmoid function
#    solution     - solution vector to the current iteration
# outputs:
#    value of the single cost iteration
def Cost_function_single_point(sigmoid_data, solution):
    # make sure arrays are not ndarrays
    solution = solution.reshape((solution.size, 1))
    sigmoid_data = sigmoid_data.reshape((sigmoid_data.size,1))
    # computer inner cost data for all K and specific datapoint
    inner_cost = np.multiply(-1*solution,np.log(sigmoid_data)) - np.multiply(
        1-solution,np.log(1-sigmoid_data))
    single_cost = np.sum(inner_cost)
    return single_cost

# Solves the cost function of a neural net for a single iteration and gradients
# for the theta values.
# input:
#    digit_data             - input data for all training examples
#    theta1_vals            - current theta 1 values array of sine KxM where K is
#                             number of hidden nodes and M is number of inputs + 1
#    theta2_vals            - current theta 2 values array of sine KxM where K is
#                             number of output nodes and M is number of hidden
#                             nodes + 1
#    digit_solutions_vec    - vecotrized solution vectors. vector solution of 1 in
#                             carrect category index and 0 everywhere else for
#                             every training example
#    lam (optional)         - lambda value to use for regularization
#    regularized (optional) - boolean value if regularization is desired
#    print_cost (optional)  - boolean value if the printing of cost to console
#                             is desired. Good inidcator for how fast neural net
#                             is being trained
#    grad_check (optional)  - Boolean calue that dictates whether the grdient
#                             values should be checked for accuracy. Only run
#                             this a few times to verify on a small neural net,
#                             then turn off
# outputs:
#    The total cost for the current theta values, as well as the theta gradients
#    between each layer
def nnCost_and_gradient_function(digit_data, theta1_vals, theta2_vals,
    digit_solutions_vec, lam = 0, regularized = False, print_cost = False,
    grad_check = False):
    # number of data features
    m = digit_data.shape[0]
    # number of output categories
    outputs = theta2_vals.shape[0]
    # number of hidden layer nodes
    hidden_nodes = theta2_vals.shape[1] - 1
    # number of input nodes
    input_nodes = theta1_vals.shape[1] - 1
    # get sigmoid formula data for each datapoint, predictions var will not be used
    predictions, h_data, hidden_layer_data, z_2_data = Predict_category_nn(
        digit_data, theta1_vals, theta2_vals)

    # solve for the gradients.
    # vectorized sigmoid gradient
    Sigmoid_gradient_vectorized = np.vectorize(Sigmoid_gradient)
    theta1_gradients = np.zeros((theta1_vals.shape))
    theta2_gradients = np.zeros((theta2_vals.shape))
    for data_segment in range(m):
        # step 1 - calculate delta 3
        delta_3 = h_data[data_segment,:] - digit_solutions_vec[data_segment,:]
        delta_3 = delta_3.reshape((delta_3.size, 1))
        # step 2 - calculate delta 2
        z_2 = (z_2_data[data_segment, :]).reshape((hidden_nodes,1))
        sigmoid_gradient = Sigmoid_gradient_vectorized(z_2)
        sigmoid_gradient = np.insert(sigmoid_gradient , 0, 1).reshape((hidden_nodes+1,1))
        delta_2 = np.multiply((theta2_vals.T @ delta_3),sigmoid_gradient)
        # step 3 - Accumalate the gradient
        a_1 = digit_data[data_segment,:]
        a_1 = np.insert(a_1,0,1).reshape((input_nodes+1,1))
        stripped_delta_2 =  delta_2[1:,:].reshape((hidden_nodes,1))
        theta1_gradients = theta1_gradients +stripped_delta_2@a_1.T
        a_2 = hidden_layer_data[data_segment,:].reshape((hidden_nodes,1))
        a_2 = np.insert(a_2, 0, 1). reshape((hidden_nodes+1,1))
        theta2_gradients = theta2_gradients + delta_3@a_2.T
    # step 4 - obtain unregularized gradient
    theta1_gradients = (1/m) * theta1_gradients
    theta2_gradients = (1/m) * theta2_gradients
    # add regularized components if desired
    if(regularized):
        # First lets do theta 1 regularized term
        theta1_regularized_term = np.zeros(theta1_gradients.shape)
        theta1_regularized_term[1:,:] = (lam/m)*theta1_vals[1:,:]
        theta1_gradients = theta1_gradients + theta1_regularized_term
        # Now theta 2 regularized term
        theta2_regularized_term = np.zeros(theta2_gradients.shape)
        theta2_regularized_term[1:,:] = (lam/m)*theta2_vals[1:,:]
        theta2_gradients = theta2_gradients + theta2_regularized_term

    if(grad_check):
        gradient_checking(digit_data, theta1_vals, theta2_vals, theta1_gradients,
            theta2_gradients, digit_solutions_vec)

    #Solve cost function in vectorized way
    cost_vals = np.zeros((m,1))
    for data_point in range(m):
        cost_vals[data_point,0] = Cost_function_single_point(
            h_data[data_point,:], digit_solutions_vec[data_point,:])
    # Cost_function_single_point_vectorized = np.vectorize(Cost_function_single_point)
    # cost_vals = Cost_function_single_point_vectorized(h_data,digit_solutions_vec)
    cost = (1/m)*np.sum(cost_vals)
    # print unregularized cost and return if desired
    if(not regularized):
        if(print_cost):
            print("Unregularized cost with given theta parameters: ", end = "")
            print(cost)
        return cost, theta1_gradients, theta2_gradients
    # calualte regularized term, but take out bias terms
    regularized_portion = (lam/(2*m)) * (((np.sum(np.square(theta1_vals)))+(
        np.sum(np.square(theta2_vals)))) - ((np.sum(np.square(theta1_vals[:,0]
        )))+(np.sum(np.square(theta2_vals[:,0])))))
    regularized_cost = cost + regularized_portion
    if(print_cost):
        print("Regularized cost with given theta parameters: ", end = "")
        print(regularized_cost)
    return regularized_cost, theta1_gradients, theta2_gradients

# Solves the cost function of a neural net for a single iteration. Same as
# Cost_function_single_point except gradients are not calculated
# input:
#    digit_data             - input data for all training examples
#    theta1_vals            - current theta 1 values array of sine KxM where K is
#                             number of hidden nodes and M is number of inputs + 1
#    theta2_vals            - current theta 2 values array of sine KxM where K is
#                             number of output nodes and M is number of hidden
#                             nodes + 1
#    digit_solutions_vec    - vecotrized solution vectors. vector solution of 1 in
#                             carrect category index and 0 everywhere else for
#                             every training example
#    lam (optional)         - lambda value to use for regularization
#    regularized (optional) - boolean value if regularization is desired
#    print_cost (optional)  - boolean value if the printing of cost to console
#                             is desired. Good inidcator for how fast neural net
#                             is being trained
# outputs:
#    The total cost for the current theta values
def Cost_checking(digit_data, theta1_vals, theta2_vals,
    digit_solutions_vec, lam, regularized = False, print_cost = False):
    # number of data features
    m = digit_data.shape[0]
    # number of output categories
    outputs = theta2_vals.shape[0]
    # number of hidden layer nodes
    hidden_nodes = theta2_vals.shape[1] - 1
    # number of input nodes
    input_nodes = theta1_vals.shape[1] - 1
    # get sigmoid formula data for each datapoint, predictions var will not be used
    predictions, h_data, hidden_layer_data, z_2_data = Predict_category_nn(
        digit_data, theta1_vals, theta2_vals)
    #Solve cost function in vectorized way
    cost_vals = np.zeros((m,1))
    for data_point in range(m):
        cost_vals[data_point,0] = Cost_function_single_point(
            h_data[data_point,:], digit_solutions_vec[data_point,:])
    # Cost_function_single_point_vectorized = np.vectorize(Cost_function_single_point)
    # cost_vals = Cost_function_single_point_vectorized(h_data,digit_solutions_vec)
    cost = (1/m)*np.sum(cost_vals)
    # print unregularized cost and return if desired
    if(not regularized):
        if(print_cost):
            print("Unregularized cost with given theta parameters: ", end = "")
            print(cost)
        return cost
    # calualte regularized term, but take out bias terms
    regularized_portion = (lam/(2*m)) * (((np.sum(np.square(theta1_vals)))+(
        np.sum(np.square(theta2_vals)))) - ((np.sum(np.square(theta1_vals[:,0]
        )))+(np.sum(np.square(theta2_vals[:,0])))))
    regularized_cost = cost + regularized_portion
    if(print_cost):
        print("Regularized cost with given theta parameters: ", end = "")
        print(regularized_cost)
    return regularized_cost

# Checks gradient values from nnCost_and_gradient_function by appoximating the
# gradient and compares
# input:
#    digit_data             - input data for all training examples
#    theta1_vals            - current theta 1 values array of sine KxM where K is
#                             number of hidden nodes and M is number of inputs + 1
#    theta2_vals            - current theta 2 values array of sine KxM where K is
#                             number of output nodes and M is number of hidden
#                             nodes + 1
#    theta1_gradients       - theta 1 gradients to check
#    theta2_gradients       - theta 2 gradients to check
#    digit_solutions_vec    - vecotrized solution vectors. vector solution of 1 in
#                             carrect category index and 0 everywhere else for
#                             every training example
#    ep (optional)          - Value for approxiamting gradient, the smaller
#                             this value is, the closer the approxiamtion gets
#                             to the actual value
# outputs:
#    No returns value, prints out the average relative difference between
#    passed in gradients and the approximated gradietns
def gradient_checking(digit_data, theta1_vals, theta2_vals, theta1_gradients,
    theta2_gradients, digit_solutions_vec, ep = 10e-6):
    # vars to hold the approxiamted gradients
    theta1_grad_check = np.zeros(theta1_gradients.shape)
    theta2_grad_check = np.zeros(theta2_gradients.shape)
    # calulate approximate gradients
    # start with theta 1 vals
    for row in range(theta1_vals.shape[0]):
        for col in range(theta1_vals.shape[1]):
            theta1_vals_plus = np.copy(theta1_vals)
            theta1_vals_plus[row,col] = theta1_vals[row,col] + ep
            theta1_vals_minus = np.copy(theta1_vals)
            theta1_vals_minus[row,col] = theta1_vals[row,col] - ep
            pos_cost = Cost_checking(digit_data, theta1_vals_plus, theta2_vals,
                digit_solutions_vec, 1)
            neg_cost = Cost_checking(digit_data, theta1_vals_minus, theta2_vals,
                digit_solutions_vec, 1)
            theta1_grad_check[row,col] = (pos_cost-neg_cost)/(2*ep)
    # now theta 2
    for row in range(theta2_vals.shape[0]):
        for col in range(theta2_vals.shape[1]):
            theta2_vals_plus = np.copy(theta2_vals)
            theta2_vals_plus[row,col] = theta2_vals[row,col] + ep
            theta2_vals_minus = np.copy(theta2_vals)
            theta2_vals_minus[row,col] = theta2_vals[row,col] - ep
            pos_cost = Cost_checking(digit_data, theta1_vals, theta2_vals_plus,
                digit_solutions_vec, 1)
            neg_cost = Cost_checking(digit_data, theta1_vals, theta2_vals_minus,
                digit_solutions_vec, 1)
            theta2_grad_check[row,col] = (pos_cost-neg_cost)/(2*ep)
    # calualte the relative differences
    theta1_rel_diffs = np.absolute(np.divide((theta1_grad_check - theta1_gradients
        ),theta1_gradients))
    theta2_rel_diffs = np.absolute(np.divide((theta2_grad_check - theta2_gradients
        ),theta2_gradients))
    average_rel_diffs = (np.sum(theta1_rel_diffs) + np.sum(theta2_rel_diffs))/(
        theta1_gradients.size + theta2_gradients.size)
    print("The average relative error was: " + str(average_rel_diffs))

# Creates initial theta value matrix
# input:
#    row           - amount of rows desired in matrix
#    col           - amount of columns desired in the matrix
#    ep (optional) - the absolute value of the max or min value that can be in
#                    in the matrix
# outputs:
#    Radomized initial theta matrix with every value between [-ep,ep], and size
#    rowxcol
def init_thetas(row, col, ep = .12):
    # use nupmy to create desired initializing theta values
    thetas = np.random.uniform(-ep,ep,(row,col))
    return thetas

# Trains neural network that has one hidden layer, and returns optimal theta
# parameters
# input:
#    data                   - data file containing features to the classification
#                             problem
#    solutions_vec          - vecotrized solution vectors. vector solution of 1 in
#                             carrect category index and 0 everywhere else for
#                             every training example
#    hidden_layer_size      - number of nodes in the hidden layer
#    a (optional)           - alpha value that effects the impact the gradient
#                             has on each value for each iteration
#    n (optinal)            - maximum number of iterations
#    lam (optinal)          - lambda value for regularization
#    regularize (optinal)   - boolean value dictating whether or not the theta
#                             values should be regularized
#    grad_limit (optional)  - the value that every gradient value must be below
#                             in order to break out of the iteration
# outputs:
#    Two 2d arrays for optimal theta values between nueral net layers
def nn_Training(data, solutions_vec, hidden_layer_size,  a = 1, n = 400, lam = 1,
    regularize = True, grad_limit = 10e-3):
    # number of data datapoints and input features
    m, num_inputs = data.shape
    # number of output categories
    categories = solutions_vec.shape[1]
    #inititalize theta 1 gradients and values
    theta1_gradients = np.ones((hidden_layer_size, num_inputs+1))
    theta1_vals = init_thetas(hidden_layer_size, num_inputs+1)
    #inititalize theta 2 gradients and values
    theta2_gradients = np.ones((categories, hidden_layer_size+1))
    theta2_vals = init_thetas(categories, hidden_layer_size+1)
    # initialize cost vars
    J_values = []
    J = np.Inf
    # iterate until optimal is found, n interations has happened, or J is
    # increasing. If n iterations happens, update alpha by multiplying it by
    # 3 to speed up process and start over. If J is increasing, alpha is too
    # large, so divide it by 3 and start over
    # optimal is found when both theta gradients are sufficiently small.
    # initialize number of iterations and theta values to 0
    iterations = 0
    while True:
        # bool to hold if we will break or not. Check theta 1 gradients
        break_out = True
        for row in theta1_gradients:
            for gradient in row:
                if np.abs(gradient) > grad_limit:
                    break_out = False
                    break
        if break_out:
            break
        # check theta 2 gradients
        break_out = True
        for row in theta2_gradients:
            for gradient in row:
                if np.abs(gradient) > grad_limit:
                    break_out = False
                    break
        if break_out:
            break

        iterations = iterations + 1
        # keep track of previous J to ensure it is not increasing
        J_prev = J
        J, theta1_gradients, theta2_gradients = nnCost_and_gradient_function(
            data, theta1_vals, theta2_vals, solutions_vec, lam, regularized
            = regularize , print_cost = True, grad_check = False)
        # check that iterations is not to high, otherwise the convergence is
        # to slow! If so, increase alpha and reset
        if(iterations > n):
            #J increased, decrease alpha, reset, and start over
            a = a*(1+random.uniform(0, 1))
            print("Trying a bigger alpha value, alpha = " + str(a), end = '\n')
            #inititalize theta 1 gradients and values
            theta1_gradients = np.ones((hidden_layer_size, num_inputs+1))
            theta1_vals = init_thetas(hidden_layer_size, num_inputs+1)
            #inititalize theta 2 gradients and values
            theta2_gradients = np.ones((categories, hidden_layer_size+1))
            theta2_vals = init_thetas(categories, hidden_layer_size+1)
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
            #inititalize theta 1 gradients and values
            theta1_gradients = np.ones((hidden_layer_size, num_inputs+1))
            theta1_vals = init_thetas(hidden_layer_size, num_inputs+1)
            #inititalize theta 2 gradients and values
            theta2_gradients = np.ones((categories, hidden_layer_size+1))
            theta2_vals = init_thetas(categories, hidden_layer_size+1)
            #reset J_values
            J_values = []
            iterations = 0
            J = np.inf
            continue
        # update thetas and J_values and scale gradients by alpha
        theta1_vals = theta1_vals - a*theta1_gradients
        theta2_vals = theta2_vals - a*theta2_gradients

        J_values.append(J)
    # print the results
    # print("The optimal theta paremeters are: " + str(repr(theta_vals)))
    # return optimat theta values
    return theta1_vals, theta2_vals

# function showing stitched image of hidden layer weights
# input:
#   theta1_vals - 2d array of weights from input layer to hidden layer
# outputs:
#    n x n stiched images of each hidden layer node weights, where n^2 is
#    amount of nodes
def Visualize_hidden_layer(theta1_vals):
    # take away bias terms, and multiply every element by 255 to make it a
    # grayscale value for the image
    theta1_vals = theta1_vals[:,1:]*255

    # final image that will have 25 hidden nodes stitched together
    final_image = np.zeros((20*5,20*5))
    # keep track of where in final_image we are storing into
    img_count = 0
    img_row = 0
    img_col = 0
    for index in range(25):
        # the digits are from 20x20 pixel images. Have to rotate 270 degrees then
        # flip along vertical axis to get desired orientation. Multpiply by 255 so
        # digits can be seen using PIL's image library
        digit = np.copy(theta1_vals[index,:]).reshape((20,20))
        final_image[img_row:img_row+20, img_col:img_col+20] = np.copy(digit)
        # update final image positioning
        img_count = img_count + 1
        img_col = img_col + 20
        if(img_count%5 == 0):
            img_row = img_row + 20
            img_col = 0
    # create and show image
    hidden_layer_image = Image.fromarray(final_image)
    hidden_layer_image.show()

def main():
    # To open the .mat files, I used the method found in this post:
    # https://stackoverflow.com/questions/874461/read-mat-files-in-python
    digit_data= scipy.io.loadmat("../Week4/machine-learning-ex3/ex3/ex3data1.mat")['X']
    digit_solutions = scipy.io.loadmat("../Week4/machine-learning-ex3/ex3/ex3data1.mat")['y']


    # vectorize the digit sultions so each each data point in a vector of 10
    # elements, with only the element coreesponding to the correct digit being 1.
    # The shape will be 5000 x 10. Used for Neural Network cost calculation.
    digit_solutions_vec = np.zeros((digit_solutions.size,10))
    # keep track of what data input we are on
    data_count = 0
    for ele in digit_solutions:
        # I use ele - 1 because the 0th element corresponds to digit 1, and the
        # 9th element to digit 0 due to octave standards
        digit_solutions_vec[data_count, ele - 1] = 1
        data_count = data_count + 1

    # show 100 images - sec 1.1
    Print_100_digits(digit_data)

    # pre calulated theta weights for forward propagation
    theta1_data = scipy.io.loadmat("../Week4/machine-learning-ex3/ex3/ex3weights.mat")["Theta1"]
    theta2_data = scipy.io.loadmat("../Week4/machine-learning-ex3/ex3/ex3weights.mat")["Theta2"]

    # This will give a unregularized cost of .287629 as it states it should in
    # ex4.pdf - sec 1.3
    nnCost_and_gradient_function(digit_data, theta1_data, theta2_data,
        digit_solutions_vec, 1, regularized = False, print_cost = True)

    # This will give a regularized cost of .383770 as it states it should in
    # ex4.pdf - sec 1.4
    nnCost_and_gradient_function(digit_data, theta1_data, theta2_data,
        digit_solutions_vec, 1, regularized = True, print_cost = True)

    # Check gradients for given theta parameters. To do this, I will create
    # a 10x10x5 neural network. 10 inputs, 8 hidden layer nodes, and 5 output
    # categories. First, I will create 100 datapoints for size of 100x10, where
    # each datapoint has a value between 10 to 1
    data_test = np.random.uniform(10, 1, (100,10))
    # create solutions to data of size 100x1 and solutions of ints [0-4]
    solutions_test = np.random.randint(0,5,(100,1))
    # create solution vectors of size 100 x 5
    solutions_vec_test = np.zeros((100,5))
    # keep track of what data input we are on
    data_count = 0
    for ele in solutions_test:
        solutions_vec_test[data_count, ele] = 1
        data_count = data_count + 1
    # create random theta 1 parameters of size 10x11
    theta1_data_test = init_thetas(10,11)
    # create random theta 2 parameters of size 5x11
    theta2_data_test = init_thetas(5,11)
    # Print Unregularized cost of thetas and check validity of gradients of
    # given neural net. The relative gradient can be seen to be very small (<1e-9),
    # confirming that our gradient calculations are correct when epsilon
    # approaches 0. - sec 2.4
    nnCost_and_gradient_function(data_test, theta1_data_test, theta2_data_test,
        solutions_vec_test, 1, regularized = False, print_cost = True,
        grad_check = True)

    # train our own neural net for digit recognition! - takes awhile, bc, duh
    theta1_vals, theta2_vals = nn_Training(digit_data, digit_solutions_vec, 25,
        a = 2, n = 10000, lam = 1, regularize = True, grad_limit = 8.75e-4)
    # see accuracy - when I ran to completion with the parameters above, I
    # obtained an acurracy of 95.04% (but depends on random initial thetas)
    # sec 2.6
    Predict_category_nn(digit_data, theta1_vals, theta2_vals,
        print_accuracy = True, solutions = digit_solutions)

    # visualize hidden layer results - sec 2
    Visualize_hidden_layer(theta1_vals)

# call main
main()
