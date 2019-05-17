#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Creates and runs an SVM model on sentence order vectors"""

__author__ = 'Haley Lepp'
__email__ = 'hlepp@uw.edu'

import os
import subprocess
import numpy as np

def create_train_file(training, feature_vector):
    """
    Parameters: training file, feature vector
    """
    count = 0
    gold = feature_vector[0]
    gold_line = ""
    for i in range(0, len(gold)):
        gold_line = gold_line + str(i + 1) + ":" + str(gold[i]) + " "
    for doc in range(1, len(feature_vector)):
        count += 1
        gold_line_1 = "1 qid:" + str(count) + " " + gold_line
        line = "2 qid:" + str(count) + " " # is this right with target?
        for i in range(0, len(feature_vector[doc])):
            line = line + str(i + 1) + ":" + str(feature_vector[doc][i]) + " "
        line = line + "\n"
        gold_line_1 = gold_line_1 + "\n"
        training.write(gold_line_1)
        training.write(line)


def create_test_file(testing, feature_vector):
    """
    Parameters: testing file, feature vector
    """
    count = 0
    for doc in range(0, len(feature_vector)):
        count += 1
        line = "0 qid:" + str(count) + " " # is this right with target?
        for i in range(0, len(feature_vector[doc])):
            line = line + str(i + 1) + ":" + str(feature_vector[doc][i]) + " "
        line = line + "\n"
        testing.write(line)


def create_svm_input(feature_vector, input_type, output_folder):
    """
    Parameters: Feature vector in which first line is gold; input_type of 'train' or 'test', name of output folder
    Outputs libSVM format files from feature vectors
    """
    # Create new folder for SVM files if needed
    svm_folder = 'src/SVM'
    folder = 'src/SVM/' + output_folder
    if not os.path.exists(svm_folder):
        os.mkdir(svm_folder)

    # Create new folder for current model if needed

    if not os.path.exists(folder):
        os.mkdir(folder)

    if input_type == 'train':
        training  = open(folder + '/training', 'w')
        create_train_file(training, feature_vector)
    elif input_type == 'test':
        testing = open(folder + '/testing', 'w')
        create_test_file(testing, feature_vector)
    else:
       raise Exception("Input type must be train or test") 


def svm_train(output_folder):
    """
    Parameters: name of output folder 
    Outputs: SVM model
    """
    folder = 'src/SVM/' + output_folder
    # run svm rank on created file
    COMMAND = '/NLP_TOOLS/ml_tools/svm/svm_rank/latest/svm_rank_learn -c 20.0 ' + folder + '/training ' + folder + '/model'
    stdout = subprocess.check_output(COMMAND.split())


def svm_test(output_folder):
    """
    Parameters: name of output folder
    Outputs file with libSVM predictions
    """
    folder = 'src/SVM/' + output_folder
    # run svm rank with existing model
    COMMAND = '/NLP_TOOLS/ml_tools/svm/svm_rank/latest/svm_rank_classify ' + folder + '/testing ' + folder + '/model ' + folder + '/output'
    # check that there is a saved model
    exists = os.path.isfile(folder + '/model')
    if exists:
        stdout = subprocess.check_output(COMMAND.split())
    else:
        raise Exception("No model file; please train model")    

# for testing
"""
if __name__ == '__main__':
    train_vectors = [[0.51851852, 0.20987654, 0.2345679 , 0.03703704],  
            [0.44444444, 0.19753086, 0.30864198, 0.04938272],
            [0.51851852, 0.20987654, 0.2345679 , 0.03703704],
            [0.43209877, 0.18518519, 0.32098765, 0.0617284 ],
            [0.41975309, 0.22222222, 0.33333333, 0.02469136],
            [0.40740741, 0.20987654, 0.34567901, 0.03703704],
            [0.40740741, 0.32098765, 0.20987654, 0.0617284 ],
            [0.30864198, 0.33333333, 0.30864198, 0.04938272],
            [0.38271605, 0.34567901, 0.2345679 , 0.03703704],
            [0.40740741, 0.34567901, 0.20987654, 0.03703704],
            [0.30864198, 0.33333333, 0.30864198, 0.04938272]]
    
   
    test_vectors = [[ 0.43055556, 0.31944444,  0.22222222,  0.02777778],
    [ 0.40277778,  0.29166667,  0.25,        0.05555556],
    [ 0.44444444,  0.30555556,  0.20833333,  0.04166667],
    [ 0.40277778,  0.29166667,  0.25,        0.05555556],
    [ 0.38888889,  0.30555556,  0.26388889,  0.04166667],
    [ 0.375,       0.31944444,  0.27777778,  0.02777778],
    [ 0.48611111,  0.26388889,  0.20833333,  0.04166667],
    [ 0.43055556,  0.26388889,  0.26388889,  0.04166667],
    [ 0.47222222,  0.27777778,  0.22222222,  0.02777778],
    [ 0.375,       0.27777778,  0.31944444,  0.02777778],
    [ 0.44444444,  0.25,        0.25,        0.05555556]]

    create_svm_input(train_vectors, 'train', 'D3')
    create_svm_input(test_vectors, 'test', 'D3')
    svm_train('D3')
    svm_test('D3')
"""   
