'''
By          : Michael Shea
Date        : 6/3/2018
Email       : mjshea3@illinois.edu
Phone       : 708 - 203 - 8272
Description :
This is the code for k-means problems. The Description of the problem can be
found in ex7.pdf part 1.
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


# function for finding the closest centroid to a point
# input:
#   data      - 1 datapoint array with it's features
#   centroids - current centroid locations
# outputs:
#    tuple consisting of the closest centroid and the corresponding distnace
def Closest_centroid(data, centroids):
    # closest centroid and corresponding distance
    closest_centroid = 0
    closest_distance = la.norm(data - centroids[0,:])
    for centroid in range(1,centroids.shape[0]):
        if(la.norm(data - centroids[centroid,:]) < closest_distance):
            closest_centroid = centroid
            closest_distance = la.norm(data - centroids[centroid,:])
    return closest_centroid, closest_distance

# function for finding the closest centroids to a dataset
# input:
#   data      - 2D array of datapoints and features
#   centroids - current centroid locations
# outputs:
#    tuple consisting of the closest centroids for each datapoints in array
#    format and the corresponding distnaces in array format
def Find_closest_Centroid(data, centroids):
    # vaar to hold the total distance each datapoint is to closest centroid
    total_dist = 0
    # define new centroids array to be returned
    data_centroids = np.zeros(data.shape[0], dtype = int)
    for data_ele in range(data.shape[0]):
        data_centroids[data_ele] , dist = Closest_centroid(data[data_ele,:], centroids)
        total_dist = total_dist + dist
    return data_centroids, total_dist

# function for computing the next locations of all centroids
# input:
#   data           - 2D array of datapoints and features
#   centroids      - current centroid locations
#   data_centroids - current centroid that each data point is closest to in
#                    array format
# outputs:
#    updated centroids array
def Compute_centroids(centroids, data, data_centroids):

    # convert data_centroids to list to be able to use count method
    data_centroids = list(data_centroids.flatten())
    # new centroid array to return
    updated_centroids = np.zeros(centroids.shape)
    # loop through data to compute mean
    for data_ele in range(data.shape[0]):
        updated_centroids[data_centroids[data_ele],:] = (
            updated_centroids[data_centroids[data_ele],:] + data[data_ele,:])
    # loop through updated centroids to finish mean calculation
    for centroid in range(updated_centroids.shape[0]):
        updated_centroids[centroid,:] = (
            updated_centroids[centroid,:] / data_centroids.count(centroid))
    return updated_centroids

# function for computing kmeans of a 2 feature dataset
# input:
#   data                      - 2D array of datapoints and features
#   num_categories            - number of categories that are desired
#   num_iterations (optional) - number of iterations to run the algorithm,
#                               where the iterations with the lowest cost
#                               will be selected
# outputs:
#    Plot data, colored by category, and the path that the best centroid location
#    took from starting point to ending point
def KMeans_2d(data, num_categories, num_iterations = 5):
    # define 2d array that wil hold the best values of the centroids
    centroids = np.zeros((num_categories, data.shape[1]))
    # array to hold categories of each element relating to above centroids
    data_centroids = np.zeros(data.shape[0])
    # run algorithm many times and choose best centroids locations
    # best cost seen so far
    best_cost = np.inf
    # for visualization purposes
    best_iteration_init = np.zeros(centroids.shape)
    for iteration in range(num_iterations):
        # current iterations optimized centroids
        iteration_centroids = np.zeros(centroids.shape)
        # random data indices that will be used to initlize centroids
        centroid_indices = random.sample(range(0, data.shape[0]), num_categories)
        for centroid in range(num_categories):
            iteration_centroids[centroid,:] = data[centroid_indices[centroid],:]
        init_centroid = np.copy(iteration_centroids)
        # perfrom k - means algo until centroids does not change
        temp_centroids = np.ones(centroids.shape)
        temp_data_centroids = np.zeros(data.shape[0])
        tot_dist = 0
        # repeat until centroids do not change
        while(la.norm(temp_centroids - iteration_centroids)/la.norm(temp_centroids) > .01):
            # for visualization purposes
            temp_centroids = np.copy(iteration_centroids)
            temp_data_centroids, tot_dist = Find_closest_Centroid(
                data,iteration_centroids)
            iteration_centroids = Compute_centroids(iteration_centroids, data,
                temp_data_centroids)
        if(tot_dist < best_cost):
            # update best centroids locations if cost is smaller
            best_cost = tot_dist
            centroids = np.copy(iteration_centroids)
            data_centroids = np.copy(temp_data_centroids)
            best_iteration_init = init_centroid

    # perform the best iteration for visualization purposed of the change in centroid
    # perfrom k - means algo until centroids does not change
    temp_centroids = np.zeros(centroids.shape)
    temp_data_centroids = np.zeros(data.shape[0])
    tot_dist = 0
    while(not (temp_centroids == best_iteration_init).all()):
        # for visualization purposes
        temp_centroids = np.copy(best_iteration_init)
        temp_data_centroids, tot_dist = Find_closest_Centroid(
            data,best_iteration_init)
        best_iteration_init = Compute_centroids(best_iteration_init, data,
            temp_data_centroids)
        # plot centroids current location
        for category in range(num_categories):
            previous_x1 = temp_centroids[category,0]
            previous_x2 = temp_centroids[category,1]
            plt.plot([previous_x1, best_iteration_init[category,0]],
                [previous_x2,best_iteration_init[category,1]], marker = 'x', color = "black")

    # visualize the data
    # supports up to 6 unique colors for differnet categories
    colors = "gbrcmyk"
    for data_ele in range(data_centroids.size):
        # plot indivdual data points with different colros for each category
        plt.scatter([data[data_ele,0]], [data[data_ele,1]], marker = 'o', color = colors[
            data_centroids[data_ele]] , facecolors = 'none')
    plt.show()


# function for computing kmeans
# input:
#   data                      - 2D array of datapoints and features
#   num_categories (optional) - number of categories that are desired
#   num_iterations (optional) - number of iterations to run the algorithm,
#                               where the iterations with the lowest cost
#                               will be selected
# outputs:
#    tuple consisting of optimal centroid locations and the coresponding centroid
#    for every data point
def KMeans(data, num_categories = 3, num_iterations = 5):
    # define 2d array that wil hold the best values of the centroids
    centroids = np.zeros((num_categories, data.shape[1]))
    # array to hold categories of each element relating to above centroids
    data_centroids = np.zeros(data.shape[0])
    # run algorithm many times and choose best centroids locations
    # best cost seen so far
    best_cost = np.inf
    for iteration in range(num_iterations):
        # current iterations optimized centroids
        iteration_centroids = np.zeros(centroids.shape)
        # random data indices that will be used to initlize centroids
        centroid_indices = random.sample(range(0, data.shape[0]), num_categories)
        for centroid in range(num_categories):
            iteration_centroids[centroid,:] = data[centroid_indices[centroid],:]
        init_centroid = np.copy(iteration_centroids)
        # perfrom k - means algo until centroids does not change
        temp_centroids = np.ones(centroids.shape)
        temp_data_centroids = np.zeros(data.shape[0])
        tot_dist = 0
        # repeat until centroids do not change
        while(la.norm(temp_centroids - iteration_centroids)/la.norm(temp_centroids) > .005):
            # for visualization purposes
            temp_centroids = np.copy(iteration_centroids)
            temp_data_centroids, tot_dist = Find_closest_Centroid(
                data,iteration_centroids)
            iteration_centroids = Compute_centroids(iteration_centroids, data,
                temp_data_centroids)
        if(tot_dist < best_cost):
            best_cost = tot_dist
            # update best centroids locations if cost is smaller
            centroids = np.copy(iteration_centroids)
            data_centroids = np.copy(temp_data_centroids)

    return centroids, data_centroids

# function for compressing an image, then reconstructing it and displaying both
# input:
#   rel_file_path             - relative file path to the image
#   num_categories (optional) - number of categories that are desired
# outputs:
#    Two side by side rgb images of original image and reconstructed image
def Img_compression_and_reconstruction(rel_file_path, num_categories = 16):
    # load image into 3d numpy array
    image = img.imread(rel_file_path)
    # turn image into Nx3 array. N - num pixels, 3 - rbg values
    image_features = np.zeros((image.shape[0] * image.shape[1], 3))
    # keep track of pixel
    pixel = 0
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            image_features[pixel,:] = np.copy(image[row,col,:])
            pixel = pixel + 1
    # obtain top number of categories for colors
    centroids, pixel_centroids = KMeans(image_features,
        num_categories = 16, num_iterations = 1)
    # make the feature matrix into a 3d rbg matrix
    image_recon = np.zeros(image.shape)
    pixel = 0
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            # mapping back from 1d matrix to 2d matrix
            image_recon[row,col,:] = centroids[pixel_centroids[image.shape[1]*row + col],:]
    # show original and reconstructed image
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(np.rot90(image,2))
    f.add_subplot(1,2, 2)
    plt.imshow(np.rot90(image_recon,2))
    plt.show(block=True)


def main():
    # data used in section 1.2
    # To open the .mat files, I used the method found in this post:
    # https://stackoverflow.com/questions/874461/read-mat-files-in-python
    mat_file_loc = "machine-learning-ex7/ex7/ex7data2.mat"
    X = scipy.io.loadmat(mat_file_loc)['X']

    # Prodcue K-means visualization on the 2d dataset for three centroidsself.
    # For section 1.2
    KMeans_2d(X, num_categories = 3)

    # produce reconstructed image - section 1.4
    # may take around 2 minutes
    rel_pic_location = "machine-learning-ex7/ex7/bird_small.png"
    Img_compression_and_reconstruction(rel_pic_location)


# call main
main()
