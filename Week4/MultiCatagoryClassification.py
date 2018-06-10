'''
By          : Michael Shea
Date        : 5/23/2018
Email       : mjshea3@illinois.edu
Phone       : 708 - 203 - 8272
Description :
This is the code for the multi class digit recognition Problem.
The Description of the problem can be found in ex3.pdf section 1.
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
# for viewing grayscale images
from PIL import Image

# Solves sigmoid formula for input z
# input:
#    z  - input to sigmoid formula
# outputs:
#    value of sigmoid formula with input z
def Sigmoid_function(z):
    # sigmoid function
    return 1/(1+np.exp(-z))

# Uses a vectorized approach to solve the cost function for classification problems
# input:
#    digit_data      - handrwritten digit pixel information. NxM Matrix, where
#                      N is the number of examples, and M is number of pixels
#    digit_solutions - Solution array to the digit_data
#    theta_vals      - current guess at theta parements for the classification
#                      problem
#    lam             - lambda value that is being used for the regression
#                      regularizations
# outputs:
#    The calculated cost and an array of values of predictions from the sigmoid
#    formula
def Vectorized_cost_function(digit_data, digit_solution, theta_vals, lam):
    # pad digit_data with ones
    digit_data_padded = np.ones((digit_data.shape[0], digit_data.shape[1]+1))
    digit_data_padded[:,1:] = np.copy(digit_data)
    # z values that go into the sigmoid function. Should be shape 400x1
    z_vals = digit_data_padded @ theta_vals.reshape(theta_vals.size,1)
    # vectorized sigmoid function
    Sigmoid_function_vectorized = np.vectorize(Sigmoid_function, otypes=[np.float])
    sigmoid_vals = Sigmoid_function_vectorized(z_vals)
    # obtain array of all the values of the cost function, vectorized.
    # First, ensure arrays are of correct dimension.
    digit_solution = digit_solution.reshape((digit_solution.size,1))
    sigmoid_vals = sigmoid_vals.reshape((sigmoid_vals.size,1))
    pre_sum_cost = np.multiply(-1*digit_solution, np.log(sigmoid_vals))-np.multiply(
        (1-digit_solution),np.log(1-sigmoid_vals))
    # regularized cost, Theta 0 is not regularized
    pre_sum_regularization = (np.sum(np.multiply(
        theta_vals[1:],theta_vals[1:])))

    # sum all cost terms and divide by training size then add regularization
    # terms to obtain complete cost
    cost = (1/digit_data.shape[0]) * np.sum(pre_sum_cost) + lam/(
        2*digit_data.shape[0])*pre_sum_regularization
    # return cost
    return cost, sigmoid_vals

# Uses a vectorized approach to caulate the theta gradients of classification
# problems
# input:
#    digit_data      - handrwritten digit pixel information. NxM Matrix, where
#                      N is the number of examples, and M is number of pixels
#    digit_solutions - Solution array to the digit_data
#    sigmoid_vals    - sigmoid function values returned from  Vectorized_cost_function
#    theta_vals      - current guess at theta parements for the classification
#                      problem
#    lam             - lambda value that is being used for the regression
#                      regularizations
# outputs:
#    Array of theta gradients to adjust the current theta values guess by
def Vectorized_gradient(digit_data, digit_solution, sigmoid_vals, theta_vals, lam):
    # pad digit_data with ones
    digit_data_padded = np.ones((digit_data.shape[0], digit_data.shape[1]+1))
    digit_data_padded[:,1:] = np.copy(digit_data)
    # First, ensure arrays are of correct dimension
    digit_solution = digit_solution.reshape((digit_solution.size,1))
    sigmoid_vals = sigmoid_vals.reshape((sigmoid_vals.size,1))
    # calulate regularized gradients using vectorization
    gradient_vals = ((1/digit_data.shape[0])* digit_data_padded.T @ (sigmoid_vals -
        digit_solution)).reshape(digit_data_padded.shape[1]) + (lam/digit_data.shape[0])*theta_vals
    # return gradients
    return gradient_vals


# function for performing logisitc regression with mutiple variable using
# regularized gradient descent. Can be used in one - vs - all or just
# normal two category classsification
# input:
#    data           - data file containing features to the classification problem
#    data_solutions - Solution array to data
#    a              - inital alpha value
#    n              - max number of iterations that the algorithm should perform
#                     before increasing its alpha value and trying again
#    lam            - lambda value that is being used for the regression
#                      regularizations
# outputs:
#    optimal theta values in an numpy array
def Two_category_classification(data, data_solution, a = 1, n = 2000, lam = 1):
    # list to hold values of cost per iteration
    J_values = []
    # m is the number of datapoints, num_features is the number of features
    m, num_features= data.shape
    # theta gradients and cost initilization. We will use 1 to start.
    # there is one more theta than features, hence the +1
    theta_gradients = np.full(num_features+1, 1)
    J = np.Inf
    # iterate until optimal is found, n interations has happened, or J is
    # increasing. If n iterations happens, update alpha by multiplying it by
    # 3 to speed up process and start over. If J is increasing, alpha is too
    # large, so divide it by 3 and start over
    # optimal is found when both theta gradients are sufficiently small.
    # initialize number of iterations and theta values to 0
    iterations = 0
    theta_vals = np.zeros(num_features + 1)
    while True:
        #bool to hold if we will break or not
        break_out = True
        for gradient in theta_gradients:
            if np.abs(gradient) > 10e-3:
                break_out = False
        if break_out:
            break
        iterations = iterations + 1
        # keep track of previous J to ensure it is not increasing
        J_prev = J
        J, sigmoid_vals =Vectorized_cost_function(data, data_solution,
            theta_vals, lam)
        theta_gradients =  Vectorized_gradient(data, data_solution,
            sigmoid_vals, theta_vals, lam)
        # check that iterations is not to high, otherwise the convergence is
        # to slow! If so, increase alpha and reset
        if(iterations > n):
            #J increased, decrease alpha, reset, and start over
            a = a*(1+random.uniform(0, 1))
            print("Trying a bigger alpha value, alpha = " + str(a), end = '\n')
            # set gradients to 1 as a temperary value
            theta_gradients = np.full(num_features+1, 1)
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
            theta_gradients = np.full(num_features+1, 1)
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
    # print("The optimal theta paremeters are: " + str(repr(theta_vals)))
    # return optimat theta values
    return theta_vals


# function for predicting the category of a multi category classification problem.
# Works with mulpitple or single datapoints, and will calculate the accuracy if
# solutions are supplied
# input:
#    data                     - data file containing features to the classification
#                               problem
#    theta_data               - data for optimized theta_values for each category.
#                               Shape of KxM, where K is number of categories
#                               and theta is number of theta values for each
#                               classification.
#    print_accuracy(optional) - State whether solutions are provided and the
#                               accuracy should be printed
#    solutions(optional)      - Solution array to data
# outputs:
#    returns an array for all the predictions and prints the accuracy of the
#    predictions if print_accuracy is true and solutions are supplied
def Predict_category(data, theta_data, print_accuracy = False, solutions = 0):
    # number of data points to predict
    num_data = data.shape[0]
    # number of categories
    num_cats = theta_data.shape[0]
    # prediction array
    predictions = np.zeros(num_data)
    # var to hold accuracy info, only usefile when print_accuracy is true
    num_true = 0
    for data_segment in range(num_data):
        # get features data padded with one for bias
        x_vals = np.ones((1, data.shape[1]+1))
        x_vals[0,1:] = np.copy(data[data_segment, :])
        # probability for each category
        cat_prob = np.zeros(num_cats)
        for cat in range(num_cats):
            theta_vals = np.copy(theta_data[cat,:]).reshape((theta_data.shape[1], 1))
            # z for sigmoid function
            z = x_vals @ theta_vals
            cat_prob[cat] = Sigmoid_function(z)
        predictions[data_segment] = np.argmax(cat_prob)
        if(print_accuracy):
            if(int(predictions[data_segment]) == solutions[data_segment]):
                num_true = num_true + 1
        # print results - ucomment if this is wanted. Used for debugging
        # print("Data for datapoint " + str(data_segment) + " is digit " + str(predictions[data_segment]))
        # if(print_accuracy):
        #     print("The correct answer was digit " + str(solutions[data_segment]))
    if(print_accuracy):
        print("Accuracy: " + str(num_true / num_data))
    # return predictions
    return predictions



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

def main():
    # To open the .mat files, I used the method found in this post:
    # https://stackoverflow.com/questions/874461/read-mat-files-in-python
    digit_data= scipy.io.loadmat("machine-learning-ex3/ex3/ex3data1.mat")['X']
    digit_solutions = scipy.io.loadmat("machine-learning-ex3/ex3/ex3data1.mat")['y']
    # the solution uses 10 for 0, lets use 0 for 0 because that makes more sense
    digit_solutions = np.array([0 if digit == 10 else digit for digit in digit_solutions])

    # show 100 images - sec 1.2
    Print_100_digits(digit_data)

    # make 10 different solution vectors for 10 digits
    solution_vectors = np.zeros((digit_solutions.size, 10))
    for digit in range(0,10):
        for ele in range(digit_solutions.size):
            if digit_solutions[ele] == digit:
                solution_vectors[ele, digit] = 1
            else:
                solution_vectors[ele, digit] = 0

    # obtain optimal theta parameters for each digit using one-vs-all.
    # +1 becuase there will be one more theta than columns of data (theta 0)
    optimal_theta_vals = np.zeros((10, digit_data.shape[1]+1))
    for digit in range(10):
        optimal_theta_vals[digit, :] = Two_category_classification(digit_data, solution_vectors[:,digit])
        print("Finished digit " + str(digit))

    # Make predictions. Note for digit recognition, accuracy is about .866
    # when stopping the regression algorithm when all gradients are less than
    # 1e-3. the 94.9% can be achieved when this value is 1e-4 to 1e-5, but the
    # algorithm takes much much longer to run (I tested it out to confirm
    # results). sec 1.3 - sec 1.4
    predictions = Predict_category(digit_data, optimal_theta_vals,
        print_accuracy = True, solutions = digit_solutions)


# call main
main()
