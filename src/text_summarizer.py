#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Nutshell, a topic-focused multi-document extractive summarization system."""

__author__ = 'Shannon Ladymon, Haley Lepp, Ben Longwill, Amina Venton'
__email__ = \
    'sladymon@uw.edu, hlepp@uw.edu, longwill@uw.edu, aventon@uw.edu'

from data_input import get_data
from content_selection import select_content
from info_ordering import order_info
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
    p.add_argument('d')
    p.add_argument('threshold')
    p.add_argument('epsilon')
    p.add_argument('output_folder')
    p.add_argument('input_list', nargs='*')
    args = p.parse_args()
    d = float(args.d)
    threshold = float(args.threshold)
    epsilon = float(args.epsilon)
    folder = str(args.output_folder)
    input_list = args.input_list


    # Read in input data
    # and return a list of Topic objects (with Documents/Sentences)

    topics = get_data(input_list)

    # Content Selection
    # identifies salient sentences & ranks them
    # & chooses up to 100 words (using full sentences)
    # Returns the list of topics with each topic.summary variable
    # modified to include a list of sentences to include

    topics_with_summaries = select_content(topics)

    # Information Ordering
    # Orders sentences by date and sentence position for each topic
    # Returns the list of topics with chronologically ordered
    # sentences in each topic.summary variable

    topics_with_summaries_in_order = order_info(topics_with_summaries)

    # Content Realization
    # Process sentences to make well-formed
    # Returns the final list of topics with well-formed sentences in each
    # topic.summary variable

    topics_with_final_summaries = realize_content(topics_with_summaries_in_order)

    # Writes summaries to file for each topic

    write_summary_files(topics_with_final_summaries)

    # Evaluates summaries for each topic
    # by running ROUGE-1 & ROUGE-2

    eval_summary(folder)


			
