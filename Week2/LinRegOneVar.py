'''
By          : Michael Shea
Date        : 5/20/2018
Email       : mjshea3@illinois.edu
Phone       : 708 - 203 - 8272
Description :
This is the code for the Linear Regression Problem with one variable. The
Description of the problem can be found in ex1.pdf section 2.
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
#    theta0 - guess for theta0
#    theta1 - guess for theta1
#    x_axis - x_axis data, or features
#    y_axis - y_axis data, or outputs
#    m      - number of samples
#    a      - alpha value for gradient descent
# outputs:
#    tuple of (in this order) J (cost), gradient for theta0, and gradient for
#    theta1. Both gradients have already been multiplied by alpha
def Cost_and_gradients_single_var(theta0, theta1, x_axis, y_axis, m, a):
    # inital gradients and cost from inital guess based from cost function
    J = 0
    theta0_grad = 0
    theta1_grad = 0
    for i in range(m):
        h_i = theta0 + theta1*x_axis[i]
        y_i = y_axis[i]
        J = J + (h_i - y_i)**2
        theta0_grad = theta0_grad + (h_i - y_i)*1 # x0 is always 1
        theta1_grad = theta1_grad + (h_i - y_i)*x_axis[i] # x1 is x_axis[i]
    J = J/(2*m)
    theta0_grad = a/m * theta0_grad
    theta1_grad = a/m * theta1_grad
    return J, theta0_grad, theta1_grad


# function for performing linear regression with one variable using gradient
# descent
# input:
#    rel_file_path         - path the csv data file
#    theta0                - initial guess for theta0
#    theta1                - initial guess for theta1
#    a                     - inital alpha value
#    n                     - max number of iterations that the algorithm should perform
#                            before increasing its alpha value and trying again
#    x_label (optional)    - x axis label of the raw data graph
#    y_label (optional)    - y axis label of the raw data graph
#    scaled (optional)    - tells whether or not the features data (x values)
#                            should be made scaled be subtracting the average
#                            and then dividing by the range of the data. Not very
#                            useful for one variable
# outputs:
#    tuple of (in this order) optimal theta1 value, optimal theta0 value, list
#    of cost values for each iteration, the average of the features, and then
#    the range of the features
def single_var_gradient_descent(rel_file_path, theta0 = 0, theta1 = 0, a = .01, n = 1000,  x_label = "", y_label = "", scaled = False):
    # initial data in case iteration has to restart due to non-convergence
    theta0_init = theta0
    theta1_init = theta1
    # init return values to 0
    x_avg = 0
    x_range = 0
    # list to hold values of cost per iteration
    J_values = []
    # assumes 0th column is x data and 1st is y data, and no header data
    with open(rel_file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        # number of samples
        m = sum(1 for row in reader)
        # put data into numpy arrays, and obtain averages and ranges
        x_avg = 0
        x_min = np.Inf # start with infinity
        x_max = np.NINF # start with neg inifinity
        x_axis = []
        y_axis = []
        # reset reader back to start
        csv_file.seek(0)
        for row in reader:
            x_cur = float(row[0])
            y_cur = float(row[1])
            x_axis.append(x_cur)
            y_axis.append(y_cur)
            x_avg = x_avg + x_cur
            if(x_cur < x_min):
                x_min = x_cur
            if(x_cur > x_max):
                x_max = x_cur
        x_axis = np.array(x_axis)
        y_axis = np.array(y_axis)
        x_avg = x_avg/m
        # range of x values
        x_range = x_max - x_min
        # make copy of original x0 data
        x_axis_orig = np.copy(x_axis)

        # make all feature data scaled
        if scaled:
            for i in range(x_axis.size):
                x_axis[i] = (x_axis[i] - x_avg)/x_range


        #theta gradients and cost initilization. We will use infinity to start
        theta0_grad = np.Inf
        theta1_grad = np.Inf
        J = np.Inf
        # iterate until optimal is found, n interations has happened, or J is
        # increasing. If n iterations happens, update alpha by multiplying it by
        # 3 to speed up process and start over. If J is increasing, alpha is too
        # large, so divide it by 3 and start over
        # optimal is found when both theta gradients are sufficiently small
        #initialize number of iterations to 0
        iterations = 0
        # the gradients are divided by alpha because they have been multiplied
        # by alpha originally
        while np.abs(theta0_grad)/a > 10e-3 or np.abs(theta1_grad)/a > 10e-3:
            iterations = iterations + 1
            # keep track of previous J to ensure it is not increasing
            J_prev = J
            J, theta0_grad, theta1_grad = Cost_and_gradients_single_var(theta0, theta1, x_axis, y_axis, m, a)
            # print(str(theta0_grad) + "    " +str(theta1_grad))
            if(J_prev < J):
                print("Trying a smaller alpha value", end = '\n')
                #J increased, decrease alpha, reset, and start over
                a = a/1.5
                # set gradients to one as a temperary value
                theta0_grad = 1
                theta1_grad = 1
                #reset thetas and interations
                theta0 = theta0_init
                theta1 = theta1_init
                #reset J_values
                J_values = []
                iterations = 0
                continue
            # check that iterations is not to high, otherwise the convergence is
            # to slow! If so, increase alpha and reset
            if(iterations > n):
                print("Trying a bigger alpha value", end = '\n')
                #increase alpha by factor of 3
                a = a*3
                # set gradients to one as a temperary value
                theta0_grad = 1
                theta1_grad = 1
                #reset thetas and interations
                theta0 = theta0_init
                theta1 = theta1_init
                #reset J_values
                J_values = []
                iterations = 0
                continue
            # update thetas and J_values
            theta0 = theta0 - theta0_grad
            theta1 = theta1 - theta1_grad
            J_values.append(J)

        # plot original raw data
        plt.subplot(221)
        plt.scatter(x_axis_orig, y_axis, marker = 'x', color = "red")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title("Raw Data")
        plt.grid(True)

        # plot the raw data with best fit line (Will be scaled if scaled is true)
        plt.subplot(222)
        plt.scatter(x_axis, y_axis, marker = 'x', color = "red")
        plt.xlabel("x0")
        plt.ylabel("Output y")
        plt.title("Raw Data with Best-Fit Line")
        plt.plot(x_axis, theta0 + theta1*x_axis)
        plt.grid(True)

        # plot Cost Fucntion to number of iterations
        plt.subplot(223)
        plt.scatter(np.arange(1,len(J_values)+1), np.array(J_values), marker = 'x', color = "red")
        plt.xlabel("# of iterations")
        plt.ylabel("Cost")
        plt.title("Cost per Iteration")
        plt.grid(True)

        # calculate and plot countour plot of theta both vlues vs cost. The
        # endpoints only really apply to the profit per city dataset
        theta0_range = np.arange(-10,10,.1)
        theta1_range = np.arange(-1,4,.025)
        theta0_values, theta1_values = np.meshgrid(theta0_range, theta1_range)
        J_values_contour = np.zeros((theta1_range.size, theta0_range.size))
        # vars x and y will be used to index the J_values_contour matrix
        x = 0
        y = 0
        for theta1_value in theta1_range:
            y = 0
            for theta0_value in theta0_range:
                # we will only be using cost return value from this function
                # this can definitely be made more effecient - will do later
                # so alpha value input does not really matter
                J_value, throw_away1, throw_away2 = Cost_and_gradients_single_var(
                    theta0_value, theta1_value, x_axis, y_axis, m, a)
                J_values_contour[x,y] = J_value
                y = y+1
            x = x + 1
        plt.subplot(224)
        #Number of contour lines
        N = 15
        plt.contour(theta0_values, theta1_values,J_values_contour, N)
        plt.xlabel("Theta0")
        plt.ylabel("Theta1")
        plt.title("Cost vs Theta0 and Theta1")
        # plot minimum theta point
        plt.plot([theta0], [theta1], marker='x', markersize=3, color="red")
        plt.grid(True)

        # main title
        if scaled:
            plt.suptitle("The Optimal Formula is y = {0:.2f} + {1:.2f}*(x0-{2:.2f})/{3:.2f}"
                .format(theta0, theta1, x_avg, x_range ))
        else:
            plt.suptitle("The Optimal Formula is y = {0:.2f} + {1:.2f}*x0"
                .format(theta0,theta1))
        #show plt figure
        plt.show()

    return theta0, theta1, J_values, x_avg, x_range



def main():
    # import data from ex1/ex1data1
    # I used this post for info :
    # https://stackoverflow.com/questions/2900035/changing-file-extension-in-python/28457540
    txt_file = "machine-learning-ex1/ex1/ex1data1.txt"
    if os.path.exists(txt_file):
        base = os.path.splitext(txt_file)[0]
        os.rename(txt_file, base + ".csv")

    # obtain theta paraments for lin regression on data - All of sections 2
    csv_data = "machine-learning-ex1/ex1/ex1data1.csv"
    theta0, theta1,J_values, x_avg, x_range = single_var_gradient_descent(csv_data,
        x_label = "Population of City in 10,000s", y_label = "Profit in $10,000s")

# call main
main()
