'''
By          : Michael Shea
Date        : 6/3/2018
Email       : mjshea3@illinois.edu
Phone       : 708 - 203 - 8272
Description :
This is the code for PCA problems. The Description of the problem can be
found in ex7.pdf part 2.
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
import matplotlib.image as img
from PIL import Image


# function for scaling a dataset
# input:
#   data - 2D array of datapoints and features
# outputs:
#    tuple consisting of scaled_data, averages for each feature, and standard
#    deviations of each feature
def Scale_data(data):
    # vars to hold the averages and standard deviations of the scaling for each
    # feature
    num_features = data.shape[1]
    # scaled data
    scaled_data = np.zeros(data.shape)
    averages = np.zeros(num_features)
    std_devs = np.zeros(num_features)
    for feature in range(num_features):
        averages[feature] = np.average(data[:,feature])
        std_devs[feature] = np.std(data[:,feature])
        scaled_data[:,feature] = (data[:,feature]-averages[feature])/std_devs[feature]
    # return scaled data and scaling features
    return scaled_data, averages, std_devs

# function performing PCA on a dataset
# input:
#   data                    - 2D array of datapoints and features
#   k (optional)            - number of features to reduce data down to
#   print_sec_22 (optional) - boolean value if the data being passed in
#                             is the data from example document 2.2
#   print_sec_23 (optional) - boolean value if the data being passed in
#                             is the data from example document 2.3
#
# outputs:
#    tuple consisting of data set with reduced features, singular values of the
#    covariance matrix, scaled data used in calculations, averages of original
#    data features, and standard deviations of original data features
def PCA(data, k = 2, print_sec_22 = False, print_sec_23 = False):
    # scale the data
    scaled_data, averages, std_devs = Scale_data(data)
    # compute covariance
    covariance = (1/data.shape[0])* scaled_data.T @ scaled_data
    # svd computation on covariance
    U,S,V = la.svd(covariance)
    # compute the reduced vectors
    U = U[:,:k]
    # print results if print_sec_22 is on
    if(print_sec_22):
        plt.scatter(scaled_data[:,0], scaled_data[:,1], color = 'b', facecolors = 'none')
        # helped by this post
        # https://stackoverflow.com/questions/42281966/how-to-plot-vectors-in-python-using-matplotlib
        plt_vects = np.array([[U[0,0], U[1,0]],[U[0,1], U[1,1]]])
        origin = [0], [0] # origin point
        # print vectors
        plt.quiver(*origin, V[:,0], V[:,1], scale=5)
        print("Top principal component: ", end = "")
        print(repr(U[:,0]))
        plt.show()
    Z = U.T @ scaled_data.T
    # print 1d representation of first original datapoint for section 3.2.1
    if(print_sec_23):
        # will print value around 1.5 like stated in example
        print(Z[0,0])
    return U, Z, scaled_data, averages, std_devs

# function to recover data from a dataset that was reduced with PCA
# input:
#   U                        - data set with reduced features
#   Z                        - singular values of the covariance matrix
#   X (optional)             - Original dataset
#   print_sec_233 (optional) - boolean value if the data being passed in
#                              is the data from example document 2.3.3, this
#                              will pring original data compared to
#                              reconstructed data
#
# outputs:
#    picture of data comparison if desired, and the appriximated reconstructed
#    data
def Recover_data(U, Z, X = None, print_sec_233 = False):
    # recvoer data
    X_approx = (U @ Z).T
    if(print_sec_233):
        if(X is not None):
            # plot connection of approximated data points and actual data points
            for data_ele in range(X_approx.shape[0]):
                x1_axis = np.array([X_approx[data_ele,0], X[data_ele,0]])
                x2_axis = np.array([X_approx[data_ele,1], X[data_ele,1]])
                plt.plot(x1_axis, x2_axis, '--', color = "black")
            # plot reconstructed data points
            plt.scatter(X_approx[:,0], X_approx[:,1], color = 'r', facecolors = 'none')
            # plot original data
            plt.scatter(X[:,0], X[:,1], color = 'b', facecolors = 'none')
            plt.show()
    return X_approx


# function showing stitched image of the first 100 faces
# input:
#   face_data - 2d array of grayscale values of all faces (must be flattened)
# outputs:
#    10 x 10 stiched images of first 100 faces
def Print_100_faces(face_data):
    # array of first 100 indices that are in the range of face samples
    rand_indices = np.arange(0,100)

    # final image that will have 100 faces stitched together
    final_image = np.zeros((32*10,32*10))
    # keep track of where in final_image we are storing into
    img_count = 0
    img_row = 0
    img_col = 0
    for index in rand_indices:
        # the faces are from 32x32 pixel images. Have to rotate 270 degrees then
        # flip along vertical axis to get desired orientation.
        face = np.flip(np.rot90(np.copy(face_data[index,:]).reshape((32,32))*1, 3), 1)
        final_image[img_row:img_row+32, img_col:img_col+32] = np.copy(face)
        # update final image positioning
        img_count = img_count + 1
        img_col = img_col + 32
        if(img_count%10 == 0):
            img_row = img_row + 32
            img_col = 0
    # create and show image
    plt.imshow(final_image, cmap='gray')
    plt.show()





def main():
    # data used in the first part of section 2
    # To open the .mat files, I used the method found in this post:
    # https://stackoverflow.com/questions/874461/read-mat-files-in-python
    mat_file_loc = "machine-learning-ex7/ex7/ex7data1.mat"
    X = scipy.io.loadmat(mat_file_loc)['X']

    # call PCA on the data - section 2.2
    PCA(X, k = 2, print_sec_22 = True)
    # section 2.3.1
    U, Z, scaled_data, averages, std_devs = PCA(X, k = 1, print_sec_23 = True)
    # recover data = section 2.3.1
    Recover_data(U,Z,scaled_data, print_sec_233 = True)

    # data used in the second part of section 2
    mat_file_loc = "machine-learning-ex7/ex7/ex7faces.mat"
    X = scipy.io.loadmat(mat_file_loc)['X']

    # print first 100 faces - section 2.4
    Print_100_faces(X)

    # print 36 most important features sec 2.4.1
    U, Z, scaled_data, averages, std_devs = PCA(X, k = 36)
    faces_approx = Recover_data(U,Z,scaled_data)
    Print_100_faces(faces_approx)

    # print 100 most important features sec 2.4.2
    U, Z, scaled_data, averages, std_devs = PCA(X, k = 100)
    faces_approx = Recover_data(U,Z,scaled_data)
    Print_100_faces(faces_approx)





# call main
main()
