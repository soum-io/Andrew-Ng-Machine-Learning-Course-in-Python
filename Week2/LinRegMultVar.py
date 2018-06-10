'''
By          : Michael Shea
Date        : 5/21/2018
Email       : mjshea3@illinois.edu
Phone       : 708 - 203 - 8272
Description :
This is the code for the Linear Regression Problem with multiple variables. The
Description of the problem can be found in ex1.pdf section 3.
'''

import numpy as np
import scipy as sp
from numpy import linalg as la
from scipy import linalg as sla
import csv
import os
import os.path
import matplotlib.pyplot as plt


# Helper function for obtaining cost and theta gradients for particular guess.
# Not very usefule by itself
# input:
#    theta_vals   - current guesses for thetas
#    x_axis       - x_axis data, or features
#    y_axis       - y_axis data, or outputs
#    m            - number of samples
#    num_features - the number of features in the dataset
#    a            - alpha value for gradient descent
# outputs:
#    tuple of (in this order) J (cost), gradients of each theta value that has
#    already been mukpltied by alpha
def Cost_and_gradients_single_var(theta_vals, x_axis, y_axis, m, num_features, a):
    # inital gradients and cost from inital guess based from cost function
    J = 0
    # one more theta than features
    theta_gradients = np.zeros(num_features+1)
    # note that I do not use linear algebra. I plan to fix this in the future
    for row in range(m):
        h_row = theta_vals[0]
        for theta in range(1,theta_vals.size):
            h_row = h_row + theta_vals[theta] * x_axis[row,theta-1]
        y_row = y_axis[row]
        J = J + (h_row - y_row)**2
        theta_gradients[0] = theta_gradients[0] + (h_row - y_row)*1 # x0 is always 1
        for feature in range(num_features):
            theta_gradients[feature+1] = theta_gradients[feature+1]+(h_row - y_row)*x_axis[row,feature]
    J = J/(2*m)
    theta_gradients = a/m * theta_gradients # element-wise multiplication
    return J, theta_gradients


# function for performing linear regression with multiple variables using gradient
# descent
# input:
#    rel_file_path         - path the csv data file
#    a                     - inital alpha value
#    n                     - max number of iterations that the algorithm should perform
#                            before increasing its alpha value and trying again
#    scaled (optional)    - tells whether or not the features data (x values)
#                            should be made scaled be subtracting the average
#                            and then dividing by the range of the data. Not very
#                            useful for one variable
# outputs:
#    tuple of (in this order) optimal theta values in an array, list
#    of cost values for each iteration, the average of the features, and then
#    the range of the features
def Mult_var_gradient_descent(rel_file_path, a = .01, n = 1000, scaled = True):
    # init return values for features. Feature one will have index one, and so
    # fourth
    x_avg = []
    x_range = []
    # list to hold values of cost per iteration
    J_values = []
    # assumes 0th column is x data and 1st is y data, and no header data
    with open(rel_file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        # number of samples
        m = sum(1 for row in reader)
        # reset reader back to start
        csv_file.seek(0)
        # number of features, -1 due to output not being a feature
        num_features = len(next(reader))-1
        # put data into numpy arrays, and obtain averages and ranges
        x_avg = np.zeros(num_features)
        x_range = np.zeros(num_features)
        x_min = np.full(num_features, np.Inf) # start with infinity
        x_max = np.full(num_features, np.NINF) # start with neg inifinity
        x_axis = np.zeros((m, num_features))
        y_axis = np.zeros(m)
        # reset reader back to start
        csv_file.seek(0)
        row_num = 0
        for row in reader:
            for col in range(num_features):
                x_cur = float(row[col])
                x_axis[row_num,col] = x_cur
                x_avg[col] = x_avg[col] + x_cur
                if(x_cur < x_min[col]):
                    x_min[col] = x_cur
                if(x_cur > x_max[col]):
                    x_max[col] = x_cur
            # num_features indexes to last element bc of zero indexing
            y_cur = float(row[num_features])
            y_axis[row_num] = y_cur
            row_num = row_num + 1
        # element wise division
        x_avg = x_avg/m
        # range of x values
        x_range = x_max - x_min
        # make copy of original x0 data
        x_axis_orig = np.copy(x_axis)
        # make x data scaled

        # make all feature data scaled
        if scaled:
            for feature in range(num_features):
                for row in range(m):
                    x_axis[row,feature] = (x_axis[row,feature] - x_avg[feature])/x_range[feature]


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
        # the gradients are divided by alpha because they have been multiplied
        # by alpha originally
        while True:
            #bool to hold if we will break or not
            break_out = True
            for gradient in theta_gradients:
                if np.abs(gradient/a) > 10e-3:
                    break_out = False
            if break_out:
                break
            iterations = iterations + 1
            # keep track of previous J to ensure it is not increasing
            J_prev = J
            J, theta_gradients = Cost_and_gradients_single_var(theta_vals, x_axis, y_axis, m, num_features, a)
            # check that iterations is not to high, otherwise the convergence is
            # to slow! If so, increase alpha and reset
            if(iterations > n):
                #J increased, decrease alpha, reset, and start over
                a = a*3
                print("Trying a bigger alpha value, alpha = " + str(a), end = '\n')
                # set gradients to 1 as a temperary value
                theta_gradients = np.full(num_features, 1)
                #reset thetas and interations
                theta_vals = np.zeros(num_features + 1)
                #reset J_values
                J_values = []
                iterations = 0
                continue
            # ensure cost is not increasing
            if(J_prev < J):
                #increase alpha by factor of 3
                a = a/1.5
                print("Trying a smaller alpha value, alpha = " + str(a), end = '\n')
                # set gradients to 1 as a temperary value
                theta_gradients = np.full(num_features, 1)
                #reset thetas and interations
                theta_vals = np.zeros(num_features + 1)
                #reset J_values
                J_values = []
                iterations = 0
                continue
            # update thetas and J_values
            theta_vals = theta_vals - theta_gradients
            J_values.append(J)


        # Uncomment to show plot Cost Fucntion vs number of iterations
        plt.scatter(np.arange(1,len(J_values)+1), np.array(J_values), marker = 'x', color = "red")
        plt.xlabel("# of iterations")
        plt.ylabel("Cost")
        plt.title("Cost per Iteration")
        plt.show(True)

        # print results
        if scaled:
            print("The Optimal Formula is y = {0:.2f}".format(theta_vals[0]), end = "")
            for feature in range(num_features):
                print("+{0:.2f}*(x{1:.0f}-{2:.2f})/{3:.2f} ".format(
                    theta_vals[feature+1], feature,x_avg[feature],x_range[feature]),end="")
            print("",end="\n")
        else:
            print("The Optimal Formula is y = {0:.2f}".format(theta_vals[0]), end = "")
            for feature in range(num_features):
                print("+{0:.2f}*x{1:.0f}".format(theta_vals[feature+1], feature),end="")
            print("",end="\n")

    return theta_vals, J_values, x_avg, x_range

# function for performing linear regression with multpiple variable using normal
# equations
# input:
#    rel_file_path         - path the csv data file
# outputs:
#    Optimal theta values in an array
def Mult_var_normal_equations(rel_file_path):
    #initilize return vars
    theta_vals = []
    # assumes 0th column is x data and 1st is y data, and no header data
    with open(rel_file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        # number of samples
        m = sum(1 for row in reader)
        # reset reader back to start
        csv_file.seek(0)
        # number of features, -1 due to output not being a feature
        num_features = len(next(reader))-1
        # use ones because first column is all ones
        x_axis = np.ones((m, num_features+1))
        y_axis = np.zeros(m)
        # reset reader back to start
        csv_file.seek(0)
        row_num = 0
        for row in reader:
            for col in range(num_features):
                x_cur = float(row[col])
                x_axis[row_num,col+1] = x_cur
            # num_features indexes to last element bc of zero indexing
            y_cur = float(row[num_features])
            y_axis[row_num] = y_cur
            row_num = row_num + 1
        y_axis = y_axis.reshape((m,1))
        # definition of normal equation
        theta_vals  = (la.pinv(x_axis.T @ x_axis)@x_axis.T@y_axis).reshape(num_features+1)
    print("The Optimal Formula is y = {0:.2f}".format(theta_vals[0]), end = "")
    for feature in range(num_features):
        print("+{0:.2f}*x{1:.0f}".format(theta_vals[feature+1], feature),end="")
    print("",end="\n")
    return theta_vals

def main():
    # import data from ex1/ex1data1
    # first need to convert .txt to .csv
    # I used this post for info :
    # https://stackoverflow.com/questions/2900035/changing-file-extension-in-python/28457540
    txt_file = "machine-learning-ex1/ex1/ex1data2.txt"
    if os.path.exists(txt_file):
        base = os.path.splitext(txt_file)[0]
        os.rename(txt_file, base + ".csv")

    # obtain theta paraments for lin regression on data
    csv_data = "machine-learning-ex1/ex1/ex1data2.csv"
    # x_avg and x_range values really only useful if unifrom was set to True
    # in the single_var_gradient_descent function call - sec 3.2
    print("Multi Variable Gradient Descent: ")
    theta_vals_grad_desc,J_values, x_avg, x_range = Mult_var_gradient_descent(csv_data, n = 1000)
    print("")
    # use normal equations method to compare - sec 3.3
    print("Multi Variable Normal Equation: ")
    theta_vals_normal = Mult_var_normal_equations(csv_data)
# call main
main()
