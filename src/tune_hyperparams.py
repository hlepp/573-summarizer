#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Script to run different hyperparameter settings on text_summarier.py"""

__author__ = 'Shannon Ladymon'
__email__ = 'sladymon@uw.edu'


from text_summarizer import summarize_text, summarize_topics_list
from data_input import get_data

import sys  #TODO: Delete

def read_results(output_folder):
    """
    TODO
    """
    results_file = 'results/' + output_folder + '_rouge_scores.out'

    with open(results_file) as f:
        for line in f:
            if "ROUGE-1 Average_R:" in line: 
                split_line = line.split()
                r1_val = float(split_line[3])
            if "ROUGE-2 Average_R:" in line: 
                split_line = line.split()
                r2_val = float(split_line[3])
        print("{}\tR-1={}\tR-2={}".format(output_folder, r1_val, r2_val))


if __name__ == '__main__':

    # TODO: Test everything
    input_path = "/dropbox/18-19/573/Data/Documents/devtest/GuidedSumm10_test_topics.xml" 

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

    # TUNED INPUT DATA PARAMETER VALUES
    lower_tuned = False # TODO: this is True when by itself, but not with stemming
    stemming_tuned = True
    idf_type_tuned = "smooth_idf" # Note this is slightly lower for config D2 & 2005, but much better for 2009
    tf_type_tuned = "term_frequency"

    # TUNED CONTENT SELECTION PARAMETER VALUES
    d_tuned_dict = {"form_D2": 0.1, "form_2005": 0.2, "form_2009": 0.3}
    summary_threshold_tuned_dict = {"form_D2": 0.4, "form_2005": 0.3, "form_2009": 0.4}
    epsilon_tuned_dict = {"form_D2": 0.02, "form_2005": 0.04, "form_2009": 0.1}
    min_sent_len_tuned_dict = {"form_D2": 3, "form_2005": 3, "form_2009": 4}

 

    # Test for data input for all configurations
    stemming_opts = [False, True] 
    lower_opts = [False, True] 
    idf_type_opts = ["smooth_idf", "probabilistic_idf", "standard_idf"]
    tf_type_opts = ["term_frequency", "log_normalization"]

    # Test for content selection for all configurations
    d_opts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    summary_threshold_opts =  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    epsilon_opts = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20] 
    min_sent_len_opts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    include_narrative = False # TODO: make this an option once added to code

    # Test only for cosine (form_D2, form_2005)
    intersent_threshold_opts =  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Test only for gen/norm (form_2009)
    mle_lambda_opts =  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    k_opts = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    ###### Get the Data #####

    topics = get_data(input_path, stemming_tuned, lower_tuned, idf_type_tuned, tf_type_tuned)


    # Run tests for each of the three formula configurations

    config_D2 = ["form_D2", "cos", "cos"]  # what we ran in D2 
    config_2005 = ["form_2005", "rel", "cos"]  # the formulas from the Otterbacher 2005 paper 
    config_2009 = ["form_2009", "gen", "norm"]  # the formulas from the Otterbacher 2009 paper
    config_list = [config_D2, config_2005, config_2009]

    for config in config_list:
        form = config[0] 
        bias_formula = config[1] 
        intersent_formula = config[2]

        # Get params tuned for each configuration
        d_tuned = d_tuned_dict[form]
        summary_threshold_tuned = summary_threshold_tuned_dict[form]
        epsilon_tuned = epsilon_tuned_dict[form]
        min_sent_len_tuned = min_sent_len_tuned_dict[form]

        # TODO: Tune narrative next, after it is added
        for min_sent_val in min_sent_len_opts:
            
            output_folder = "D3_tune_" + form + "_min_sent_" + str(min_sent_val)

            summarize_topics_list(topics, output_folder, d_tuned, intersent_threshold_default, summary_threshold_tuned, epsilon_tuned, mle_lambda_default, k_default, min_sent_len_tuned, include_narrative_default, bias_formula, intersent_formula)

            read_results(output_folder)

        print()  # Print newline to separate

