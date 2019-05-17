#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Creates and runs an SVM model on sentence order vectors"""

__author__ = 'Haley Lepp'
__email__ = 'hlepp@uw.edu'

import os
import subprocess


def create_svm_input(feature_vector, input_type):
    """
    Parameters: Feature vector in which first line is gold; input_type of 'train' or 'test'
    Outputs libSVM format files from feature vectors
    """
    # create new file
    if not os.path.exists('src/SVM'):
        os.mkdir('src/SVM')
    if input_type == 'train'
        training  = open('src/SVM/training', 'w')
    elif input_type == 'test'
        testing = open('src/SVM/testing', 'w')
    else:
       raise Exception("Input type must be train or test") 
    # print feature vectors into file
    count = 0
    for doc in feature_vector:
        if count == 0:
            line = "GOLD "
        else:
            line = "PERM "
	count += 1
        for i in range(0, len(doc)):
            line = line + i + ":" + doc[i] + " "
        line = line + "\n"
        if input_type == 'train':
            training.write(line)
        else: 
            testing.write(line)


def svm_train(t, g, d, r):
    """
    Parameters: libSVM: kerneltype, gamma, degree, coef0
    Outputs: libSVM model
    """
    # run svm light on created file
    COMMAND = 'svm-train -'  + t + " -" + g + " -" + d + " -" + r + ' src/SVM/training src/SVM/model'
    subprocess.call(COMMAND)


def svm_test()
    """
    Parameters:
    Outputs file with libSVM predictions
    """
    # run svm light with existing model
    COMMAND = 'svm-predict src/SVM/testing src/SVM/model src/SVM/output'
    # check that there is a saved model
    exists = os.path.isfile('src/SVM/model')
    if exists:
        subprocess.call(COMMAND)
    else:
        raise Exception("No model file; please train model")    
    
