'''
By          : Michael Shea
Date        : 5/30/2018
Email       : mjshea3@illinois.edu
Phone       : 708 - 203 - 8272
Description :
This is the code for solving email spam with SVM. The Description of the problem
can be found in ex6.pdf part 1.
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
import re
from stemming.porter2 import stem
import string

# function for turning the body of an email into filtered text
# input:
#   text                 - original email as string
#   print_text (optinal) - boolean that says if the original and filtered email should
#                          be printed
# outputs:
#    filtered email as string
def Filter_email(text, print_text = False):
    if(print_text):
        print(text)
    # make lowercase
    text = text.lower()
    # strip html tags. From this post -
    # https://stackoverflow.com/questions/753052/strip-html-from-strings-in-python
    text = re.sub(r'<[^<>]+>', ' ', text)
    # normilze numbers
    text = re.sub(r'[0-9]+', 'number', text)
    # normalize urls from this post -
    # https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
    text = re.sub(r'(http|https)://[^\s]*', 'httpaddr', text)
    # normilize email address.
    text = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', text)
    # handle $ sign
    text = re.sub(r'[$]+', 'dollar', text)
    # get rid of punctiation
    text = text.translate(string.punctuation)
    # Remove any non alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    # stem text
    text = " ".join([stem(word) for word in text.split(" ")])
    # make all white space single space
    text = re.sub(r'\s+', ' ', text)
    if(print_text):
        print(text)
    # return filtered text
    return text

# function that turns email into array of indices to common words that it
# contains
# input:
#   email                - original email as string
#   vocab_dict           - dictionary of word to it's indeces in the list of common words
#   print_text (optinal) - boolean that says if the original and filtered email should
#                          be printed
# outputs:
#    array of common words indeces contained in email
def Text_to_Vec(email, vocab_dict, print_text = False):
    filtered_email = Filter_email(email)
    # create list that will hold indices of vocab words in the email
    text_vec = []
    for word in filtered_email.split(" "):
        if word in vocab_dict.keys():
            text_vec.append(vocab_dict[word])
    if(print_text):
        print(repr(text_vec))
    return text_vec

# function that turns filtered email into feature vector of 0's and 1's corresponding
# to if it contains the word in the common words array at each features indice of the
# input:
#   email                - original email as string
#   vocab_dict           - dictionary of word to it's indeces in the list of common words
#   print_text (optinal) - boolean that says if the original and filtered email should
#                          be printed
# outputs:
#    feature vector for spam email classification
def Email_features(email, vocab_dict, print_indices = False):
    word_indeces = Text_to_Vec(email, vocab_dict)
    # vector to hold input features of email
    feature_vector = np.zeros(len(vocab_dict))
    for index in word_indeces:
        # -1 due to 1 indexing it matlab and to match the example
        feature_vector[index-1] = 1
    if(print_indices):
        print(repr(feature_vector))
    return feature_vector

# function for solving model using SVM.
# input:
#   X                 - Training data
#   y                 - training solution
#   C (optional)      - C value in SVM algorithm, used for regularization
#   sig (optional)    - sig value used in gaussian elimination
#   kernel (optional) - type of SVM kernel to use
#   graph (optional)  - If the data orginally has two variables, graph the
#                       classifiers output
# outputs:
#    Weight values (except for gaussian elimination), and classifier
def SVM_solve(X, y, C = 1, sig = 1, kernel = "linear"):
    # define svm for the classifier. SVC is scikit learn's SVM with C param
    # support
    if(kernel == "gaussian"):
        # 'rbf' kernel is same as gaussian - stands for 'Radial basis function'
        clf = SVC(C = C, kernel = "rbf", gamma = sig)
        # train model on data
        # ensure y is correct shape
        y = y.reshape(y.size)
        clf.fit(X, y)
        # return classifier
        return clf
    else:
        clf = SVC(C = C, kernel = kernel)
        # train model on data
        # ensure y is correct shape
        y = y.reshape(y.size)
        clf.fit(X, y)
        # obtain optimized theta parameters
        theta_vals = np.zeros(X.shape[1]+1)
        theta_vals[0] = clf.intercept_
        theta_vals[1:] = clf.coef_
        # return optimal theta parameters and classifier
        return theta_vals, clf


def main():
    # import vocab data from ex6/vocab.txt
    vocab_path = "machine-learning-ex6/ex6/vocab.txt"
    vocab = []
    with open(vocab_path, 'r') as file:
        # create list of all striped lines from text file
        vocab = sum([l.strip().split(' ') for l in file], [])
    for word in range(len(vocab)):
        # remove all none alpha chars
        vocab[word] = ''.join([i for i in vocab[word] if i.isalpha()])

    # create dictionary of vocab terms
    vocab_dict  = {}
    for index, word in enumerate(vocab):
        # +1 because it will match closer to excersize since octave is 1 indexed
        vocab_dict[word] = index+1

    # test Filter_email function - will not produce exactly the same results
    # as shown in ex6.pdf, but very similiar
    test = "> Anyone knows how much it costs to host a web portal ?\n >\n Well, it depends on how many visitors you're expecting.\n This can be anywhere from less than 10 bucks a month to a couple of $100.\n You should checkout http://www.rackspace.com/ or perhaps Amazon EC2\n if youre running something big..\n To unsubscribe yourself from this mailing list, send an email to:\n groupname-unsubscribe@egroups.com"
    Filter_email(test, print_text = True)

    # Vectorize example secion 2.1.1
    Text_to_Vec(test, vocab_dict, print_text = True)

    # Input vector example section 2.2
    Email_features(test, vocab_dict, True)

    # data used in section 2.3
    # To open the .mat files, I used the method found in this post:
    # https://stackoverflow.com/questions/874461/read-mat-files-in-python
    # training data
    mat_file_loc = "machine-learning-ex6/ex6/spamTrain.mat"
    Xtrain = scipy.io.loadmat(mat_file_loc)['X']
    ytrain = scipy.io.loadmat(mat_file_loc)['y']
    # testing data
    mat_file_loc = "machine-learning-ex6/ex6/spamTest.mat"
    Xtest = scipy.io.loadmat(mat_file_loc)['Xtest']
    ytest = scipy.io.loadmat(mat_file_loc)['ytest']

    # obtain classifier and weights
    theta_vals, clf =  SVM_solve(Xtrain, ytrain, C = .01, sig = 1, kernel = "linear")
    train_predict = clf.predict(Xtrain)
    test_predict = clf.predict(Xtest)
    # calculate accuracies
    train_accuracy = 0
    test_accuracy = 0
    # loop through training predictions
    for index in range(train_predict.size):
        if(train_predict[index] == ytrain[index]):
            train_accuracy = train_accuracy + 1
    train_accuracy = train_accuracy/(train_predict.size)
    # loop through testing predictions
    for index in range(test_predict.size):
        if(test_predict[index] == ytest[index]):
            test_accuracy = test_accuracy + 1
    test_accuracy = test_accuracy/(test_predict.size)
    # print results - will be slightly different than examples due to different
    # implementations of svm
    print("Training accuracy: " + str(train_accuracy))
    print("Test accuracy: " + str(test_accuracy))

    # sorted array of indices of the weights - section 2.4
    sorted_indices = np.fliplr([np.argsort(theta_vals)])[0]
    # print top words
    for i in range(15):
        # don't print if weight is bias term
        if(sorted_indices[i] == 0):
            continue
        # -1 due to the 0th term being a bias
        print(str(vocab[sorted_indices[i]-1]) + " ", end = "")

    # section 2.5 - test of sample emails
    # non spam example
    with open('machine-learning-ex6/ex6/emailSample1.txt', 'r') as file:
        sample_email = file.read().replace('\n', '')
    sample_features = Email_features(sample_email, vocab_dict)
    # print result - should print 0 as it is not spam
    print(clf.predict(sample_features.reshape((1,sample_features.size))))
    # spam example
    with open('machine-learning-ex6/ex6/spamSample1.txt', 'r') as file:
        sample_email = file.read().replace('\n', '')
    sample_features = Email_features(sample_email, vocab_dict)
    # print result - should print 1 as it is spam
    print(clf.predict(sample_features.reshape((1,sample_features.size))))



# call main
main()
