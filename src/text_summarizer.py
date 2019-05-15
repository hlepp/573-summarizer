#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Nutshell, a topic-focused multi-document extractive summarization system."""

__author__ = 'Shannon Ladymon, Haley Lepp, Ben Longwill, Amina Venton'
__email__ = \
    'sladymon@uw.edu, hlepp@uw.edu, longwill@uw.edu, aventon@uw.edu'

from data_input import get_data, get_gold_standard_docs
from content_selection import select_content
from info_ordering import order_info_chron, order_info_entity, get_training_vectors
from content_realization import realize_content
from evaluation import eval_summary
from sys import argv
import argparse
import os


def write_summary_files(topics_with_final_summaries):
    """
    This function takes in a list of topics with summaries
    generated by realize_content(topics_with_summaries_in_order)
    in text_summarizer.py main.
    It outputs a file for each topic with the final summary.
    """

    # Directory where output files should be

    output_dir = 'outputs/' + folder + '/'

    # Variable to create unique ending for files

    # TODO: change to be the folder name
    # once ROUGE can process that output
    numeric_count = 1

    for topic in topics_with_final_summaries:

        # Get topic id for each Topic object
        topic_id = topic.topic_id 
       
        # Split topic ID into 2 parts
        id_part1 = topic_id[:-1]
        id_part2 = topic_id[-1:]

        # Make output file name and directory
        file_path = os.path.join(output_dir
                                 + '{}-A.M.100.{}.{}'.format(id_part1,
                                 id_part2, str(numeric_count)))
        directory = os.path.dirname(file_path)

        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Write sentences to topic output file
        with open(file_path, 'w') as out_file:

            # Get list of summary Sentence objects from the Topic object
            summary_sentences = topic.summary
            
            for sentence in summary_sentences:
                out_file.write(sentence.original_sentence + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('data_type')
    p.add_argument('d')
    p.add_argument('threshold')
    p.add_argument('epsilon')
    p.add_argument('output_folder')
    p.add_argument('input_file')
    args = p.parse_args()
    data_type = args.data_type # Either train, eval, or dev
    d = float(args.d)
    threshold = float(args.threshold)
    epsilon = float(args.epsilon)
    folder = str(args.output_folder)
    input_path = str(args.input_file)



    # Read in input data
    # and return a list of Topic objects (with Documents/Sentences)

    topics = get_data(input_path,stemming=False,lower=False,idf_type='smooth_idf')

    # Content Selection
    # identifies salient sentences & ranks them
    # & chooses up to 100 words (using full sentences)
    # Returns the list of topics with each topic.summary variable
    # modified to include a list of sentences to include

    topics_with_summaries = select_content(topics)

    #TODO: where should training for entity grids sentence ordering go
    # Training for Information Ordering
    # Read in gold summary document data
    # Return a list of document objects
    
    gold_summ_path = "/dropbox/18-19/573/Data/models/training/2009"
    gold_summ_docs = get_gold_standard_docs(gold_summ_path)
    #TODO: add path to condor file for argparse
    
    # Generates entity grids and feature representations 
    # for all document permutations
    # Returns the list of feature representations in 
    # 2D numpy arrays for each document
    all_docs_vectors = get_training_vectors(gold_summ_docs)

    #TODO: Modify stub for ordering with entity grid model
    # Information Ordering
    # Entity-Based Approach
    # Orders sentences by best ranked from entity grid model for each topic
    # Returns the list of topics with optimally ordered
    # sentences in each topic.summary variable

    # topics_with_summaries_in_order = order_info_entity(topics_with_summaries)

    # Information Ordering
    # Chronological Order Approach
    # Orders sentences by date and sentence position for each topic
    # Returns the list of topics with chronologically ordered
    # sentences in each topic.summary variable

    topics_with_summaries_in_order = order_info_chron(topics_with_summaries)

    # Content Realization
    # Process sentences to make well-formed
    # Returns the final list of topics with well-formed sentences in each
    # topic.summary variable

    topics_with_final_summaries = realize_content(topics_with_summaries_in_order)

    # Writes summaries to file for each topic

    write_summary_files(topics_with_final_summaries)

    # Evaluates summaries for each topic
    # by running ROUGE-1 & ROUGE-2

    eval_summary(folder, data_type)


			
