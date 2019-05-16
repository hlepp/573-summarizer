#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Script to run different hyperparameter settings on text_summarier.py"""

__author__ = 'Shannon Ladymon'
__email__ = 'sladymon@uw.edu'


#from text_summarizer import summarize_text

# TODO: Change text_summarizer to have a method, which is called by its main
# (rather than everything in main as it is now
# Possibly also rename to "nutshell.py"??


if __name__ == '__main__':

    # TODO: Test everything
    input_path = "/dropbox/18-19/573/Data/Documents/devtest/GuidedSumm10_test_topics.xml" # TODO:add train/eval as options

    # D2 defaults
    stemming_default = False
    lower_default = False
    include_narrative_default = False
    idf_type_default = "smooth_idf"
    tf_type_default = "term_frequency"
    d_default = 0.7
    summary_threshold_default = 0.5
    epsilon_default = 0.1
    min_sent_len_default = 5
    intersent_threshold_default = 0.0

    # 2009 defaults
    mle_lambda_default = 0.6
    k_default = 20


    # Test for all formulas
    stemming_opts = [False, True] 
    lower_opts = [False, True] 
    include_narrative = False # TODO: make this an option once added to code
    idf_type_opts = ["smooth_idf", "probabilistic_idf", "standard_idf"]
    tf_type_opts = ["term_frequency", "log_normalization"]
    d_opts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    summary_threshold_opts =  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    epsilon_opts = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20] 
    min_sent_len_opts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Test only for cosine (form_D2, form_2005)
    intersent_threshold_opts =  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Test only for gen/norm (form_2009)
    mle_lambda_opts =  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    k_opts = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    # Run tests for each of the three formula configurations

    ###### form_D2 - what we ran in D2 ######
    form = "form_D2"
    bias_formula = "cos"
    intersent_formula = "cos"

    # Tune only stemming
    for val in stemming_opts:
        output_folder = "D3_tune_" + form + "_stemming_" + str(val)

        # TODO: update this after seeing what other params people add
#        summarize_text(input_path, output_folder, stemming=val, lower=lower_default, idf_type=idf_type_default, , tf_type=tf_type_default, d=d_default, intersent_threshold=intersent_threshold_default, summary_threshold=summary_threshold_default, epsilon=epsilon_default, mle_lambda=mle_lambda_default, k=k_default, min_sent_len=min_sent_len_default, include_narrative=include_narrative_default, bias_formula, intersent_formula)


    ###### form_2005 - the formulas from the Otterbacher 2005 paper ######
    form = "form_2005"
    bias_formula = "rel"
    intersent_formula = "cos"

    # Tune only stemming
    for val in stemming_opts:


    ###### form_2009 - the formulas from the Otterbacher 2009 paper ######
    form = "form_2009"
    bias_formula = "gen"
    intersent_formula = "norm"

    # Tune only stemming
    for val in stemming_opts:
        output_folder = "D3_tune_" + form + "_stemming_" + str(val)

