#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from operator import itemgetter
import numpy as np
import itertools



"""Information ordering for multi-document text summarization. Two different approaches: entity-based & chronological order"""

__author__ = "Amina Venton"
__email__ = "aventon@uw.edu"



def get_entity_grids(sentence_list, doc_nouns_list):
    """
    This function takes in a list of sentence tuples (sent_pos, sent_nouns) 
    and list of document nouns and creates an entity grid of the gold document 
    
    Then, it generates and returns 11 entity grid permutations of the document
    """
    
    # List to hold the original gold summary grid
    # The grid will become n * m 
    # Where n = num of sents in doc & m = num of nouns in the doc
    gold_entity_grid = []

    for sentence in sentence_list:
        
        # Get individual sentence nouns
        (sent_pos, sent_nouns) = sentence
        
        # List to store entity
        sent_row = []

        # Fill out the gold entity grid
        # Nouns for each sentence either: 
        # 1 = present 0 = absent
        for noun in doc_nouns_list:
            
            if noun in sent_nouns:
                sent_row.append(1)
            else:
                sent_row.append(0)

        # Add sentence row of entities to grid
        gold_entity_grid.append(sent_row)

    
    #TODO: Test various # of permutations as a param? Memory issue
    # Get a list of 11 possible sentence orderings in the document from original entity grid
    # The first permutation in the list is the original gold standard sentence ordering
    entity_grid_permutations = [ list(doc_perm_tuple) for doc_perm_tuple in itertools.islice(itertools.permutations(gold_entity_grid), 11) ]

    # Return numpy array of 10  n * m entity grid permutations for the document
    return np.array(entity_grid_permutations)


def get_doc_vectors(entity_grid_permutations):
    """
    This function takes in a list of entity grid permutations and generates 
    a feature vector representation of each grid
   
    A list of documennt feature representations is returned
    """
   
    # List to store all transition probability vectors
    # First vector is the original gold standard sentence ordering
    doc_vectors = []

    # Iterate through each permutation
    for entity_grid in entity_grid_permutations:

        # Create a sequences dictionary to keep track of counts
        sequences_dict = {"00" : 0, "01" : 0, "10" : 0, "11" : 0}

        # Tally up sequences vertically in each grid
        # Transpose entity grid to get sequences horizontally instead
        entity_grid_transpose = entity_grid.transpose()

        # Tally seqs in each row
        for row in entity_grid_transpose:
            for i in range(1, entity_grid_transpose.shape[1]):
                 seq = str(row[i-1]) + str(row[i])
                 sequences_dict[seq] = sequences_dict[seq] + 1

        
        # Create an ordered sequence list for all vectors
        # ["00", "01", "10", "11"]
        sequence_list = sorted(sequences_dict.keys())
        
        # Get the total transitions, it should be the same for each vector of the document
        total_transitions = sum(sequences_dict.values())

        # Get feature vector representation of each grid
        # Dividing by the total number of transitions will be done last in one step
        feat_vector = []

        for seq in sequence_list:
            feat_vector.append(sequences_dict[seq])
        
        # Add each feature vector to the list of document vectors
        doc_vectors.append(np.array(feat_vector))

    # Divide doc vectors by the total number of transitions
    doc_vectors = np.array(doc_vectors) / total_transitions

    # Return list of document vectors
    return doc_vectors


def get_doc_data(gold_summ_docs):
    """
    This function takes in a list of gold summary Document objects
    and returns a dictionary of the data.

    Keys are document object and values are list of tuples of (sent, set of nouns) 
    for each sent in the doc. {doc_obj: [(sent_pos, sent_noun_set)]
    """
    
    doc_dict = dict()

    for summary_doc in gold_summ_docs:
        
        # List to hold (sent_position, sent_noun_set) tuples
        doc_list = []

        for sent_obj in summary_doc.sentence_list:
            doc_list.append((sent_obj.index, sent_obj.nouns))
        
        # Add the list of sentence tuples to the dictionary
        doc_dict[summary_doc] = doc_list
    
    # Return the dictionary
    return doc_dict


def get_training_vectors(gold_summ_docs):
    """
    This function takes in a list of gold summary Document objects
    and makes entity grids and feature vector representations 
    for each summary document.

    It returns a list of 2D numpy arrays where each array consists of 
    the gold summary feature vector and all of its vector permutations.
    """

    # Master list of 2D feature vector permuations of each document
    all_training_vectors = []
    
    # Get a dictionary of document data
    doc_dict = get_doc_data(gold_summ_docs)

    # Iterate through all the documents to make grids and vectors
    for document in doc_dict:
    
        # Get list of all nouns for each document
        all_nouns_set = set()
        all_nouns_sets = [all_nouns_set.union(sent_noun_set) for (sent_pos, sent_noun_set) in doc_dict[document]]
        doc_nouns_list = sorted(list(set.union(*all_nouns_sets)))

        # Get entity grids for (11) permutations of the document
        entity_grid_permutations = get_entity_grids(doc_dict[document], doc_nouns_list)

        # Get probability transition vectors and feature vector representations for each 
        # permutation of the document
        doc_vectors = get_doc_vectors(entity_grid_permutations)
        
        # Append numpy feature vectors to master list of document vectors
        all_training_vectors.append(doc_vectors)

    # Return list of all feature vectors
    all_training_vectors


def build_entity_model(output_folder):
    pass

def order_info_entity(topics_with_summaries, num_permutations, output_folder):
    """
    Entity-based ordering approach
    This function takes in a list of Topic objects with ranked summaries
    generated by select_content(topics) in text_summarizer.py main.

    It uses a coherence model trained on gold summary data to find 
    the best ordering of the ranked summaries. 
    """
    
    #TODO: Get feature vector representations of each summary and feed into model
   
    # Get the list of ranked Sentence objects per Topic object
    # Get sentence data for each summary
    # Use functions to create grid and feat vector permutations
    # Feed vectors into model
    # Get the best ordering
    # Return list of Topics with ordered sentences in the Topic summary

    # For now, return original ranked summaries
    return topics_with_summaries


def order_info_chron(topics_with_summaries):
    """
    Chronological ordering approach
    This function takes in a list of Topic objects with ranked summaries
    generated by select_content(topics) in text_summarizer.py main.
    
    It orders the Sentence objects chronologically (earliest date first) 
    for each Topic summary and returns a list of Topics with chronologically 
    ordered summaries
    """

    # Sort original dates in order and then by sentence position
    for topic in topics_with_summaries:
        
        # Get the list of ranked Sentence objects per Topic object
        sentences = topic.summary

        # Create list of tuples of Sentence objects with their date and position
        # [(sentence_object, sentence_date, sentence_position)]
        sentence_tuples = [(sentence_object, sentence_object.parent_doc.date, sentence_object.index) for sentence_object in sentences]
        
        # Sort Sentence objects first by date and then by position in the doc
        sentence_tuples.sort(key=itemgetter(1,2))

        # Get only the Sentence object from each tuple
        sorted_sent = [sen_tup[0] for sen_tup in sentence_tuples]

        # Add chronologically sorted Sentence objects list for each Topic object summary
        topic.summary = sorted_sent
        
    # Return list of Topics with chronologically ordered sentences in the Topic summary
    return topics_with_summaries
