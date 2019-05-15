#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Script to run different hyperparameter settings on text_summarier.py"""

__author__ = 'Shannon Ladymon'
__email__ = 'sladymon@uw.edu'


import text_summarizer

# TODO: Change text_summarizer to have a method, which is called by its main
# (rather than everything in main as it is now
# Possibly also rename to "nutshell.py"??


if __name__ == '__main__':

    # TODO: Test everything
    input_path = "/dropbox/18-19/573/Data/Documents/devtest/GuidedSumm10_test_topics.xml" # TODO:add train/eval as options
    output_folder = "D3"  # change the name for EACH RUN
    stemming = False # False/True
    lower = False # False/True
    idf_type = "smooth_idf"  # TODO: Get all options for this
    d = 0 #0-1
    intersent_threshold = 0 # 0-1  TODO: check the range of non-cosine values
    summary_threshold = 0 # 0-1
    epsilon = 0.01 # TODO: Look up valid values of epsilon
    mle_lambda = 0 # 0-1
    include_narrative = False # False/True
    min_sent_len = 0 #0-10?
    bias_formula = "cos"  # "cos", "rel", "gen"
    intersent_formula = "cos"  # "cos", "norm"

    # TODO: test value ranges for the following (do we need to check interactions? What settings for the other things?)
    # d
    # intersent_threshold
    # summary_threshold
    # epsilon
    # mle_lambda
    # min_sent_len

    # EXAMPLE of how to run these - TODO: make for each hyperparam to test (nest? don't nest?)
    for val in range(0,1, 0.1):
        d = val
        output_folder += ".d." + val 
        # TODO: update this after seeing what other params people add
        summarize_text(input_path, output_dir, stemming, lower, idf_type, d, intersent_threshold, summary_threshold, epsilon, mle_lambda, include_narrative, min_sent_len, bias_formula, intersent_formula)

