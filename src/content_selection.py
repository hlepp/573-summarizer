#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Content Selector for multi-document text summarization that ranks and chooses the sentences to use in the summary."""

__author__ = "Shannon Ladymon, Haley Lepp, Amina Venton"
__email__ = "sladymon@uw.edu, hlepp@uw.edu, aventon@uw.edu"

from data_input import Topic, Document, Sentence, Token
import math
import numpy as np


def _cosine_similarity(sentence_1, sentence_2):
    """
    returns cosine similarity of two sentence objects
    """
    numerator = 0
    denominator_1 = 0
    denominator_2 = 0
    for i in sentence_1.tf_idf.keys():
        numerator += sentence_1.tf_idf.get(i) * sentence_2.tf_idf.get(i, 0.0)
        denominator_1 += sentence_1.tf_idf.get(i) ** 2
    for i in sentence_2.tf_idf.values():
        denominator_2 += i * i
    denominator = math.sqrt(denominator_1 * denominator_2)
    if denominator != 0:
        return numerator / denominator
    else:
        return denominator  


def _build_sim_matrix(sent_list, threshold):
    """
    Builds and returns a 2D numpy matrix of inter-sentential cosine similarity.
    """
    num_sent = len(sent_list)

    # [num_sent] x [num_sent] matrix, defaulting to 0 for each similarity
    sim_matrix = np.zeros((num_sent, num_sent))

    # Fill the 2D numpy matrix with inter-sentential cosine similarity
    for i in range(num_sent):

        # only iterate over top half triangle of matrix
        # since bottom half is identical
        for j in range(i, num_sent):

            # Get the cosine similarity for different sentences
            if i != j:
                sim = _cosine_similarity(sent_list[i], sent_list[j])

		# Only include similarities above the threshold
                if sim >= threshold:
                    sim_matrix[i][j] = sim
                    sim_matrix[j][i] = sim
            else:
                # If the same sentence, sim is 1.0
                sim_matrix[i][i] = 1.0


    # Normalize by dividing by sum of each row
    row_sums = sim_matrix.sum(axis=1, keepdims=True)

    sim_matrix = sim_matrix / row_sums

    return sim_matrix


def _build_bias_vec(sent_list, topic_sent, include_narrative = False, bias_formula = 0):
    """
    Builds and returns a 1D numpy vector of the similarity between each sentence
    and the topic title.
    """

    # Holds the similarity for each sentence with the topic descritpion
    num_sent = len(sent_list)

    #1D matrix to hold the similiarity between each sentence and the topic
    bias_vec = np.zeros(num_sent) 

    # Get similarity for each sentence and the topic
    for i in range(num_sent):

        # Use the specified bias formula
        if bias_formula == 0:
            bias_vec[i] = _cosine_similarity(sent_list[i], topic_sent)
        else:
            topic = topic_sent.parent_doc
            topic_idf_dict = topic.idf
            bias_vec[i] = _calc_relevance(sent_list[i], topic_sent, topic_idf_dict)

    # Normalize by dividing by the sum
    bias_sum = np.sum(bias_vec)

    # If the bias sum is 0, change to 1 to avoid division by 0
    if bias_sum == 0:
        bias_sum = 1

    bias_vec = bias_vec / bias_sum

    return bias_vec


def _calc_relevance(sent, topic_sent, topic_idf_dict):
    "Calculates relevance for two sentences."
    # TODO: Test that this works

    rel_sum = 0

    # TODO: fix to devide by idf
    for i in sent.tf_idf.keys():
        rel_sum += math.log(sent.tf_idf.get(i) + 1) * math.log(topic_sent.tf_idf.get(i, 0.0) + 1) * topic_idf_dict.get(i) 

    return rel_sum


def _build_markov_matrix(sim_matrix, bias_vec, d):
    """
    Builds and returns the markov matrix (matrix to multiply by in the power method)
    using the Biased LexRank formula with cosine similarity.
    """
    markov_matrix = (d * (bias_vec)) + ((1-d) * (sim_matrix))
    return markov_matrix


def _power_method(markov_matrix, epsilon):
    """
    Uses the power method to find the LexRank values of each sentence
    and returns a 1D vector of the final LexRank values after convergence.
    """

    num_sent = len(markov_matrix)

    # Transpose the markov_matrix for power method calculations
    transition_matrix = markov_matrix.T 

    # Create an original vector of sentence probabilies
    # using uniform distribution
    prob_vec = np.ones(num_sent)/num_sent

    # Iterate through power method until convergence
    matrix_diff = 1.0
    while matrix_diff > epsilon:

        # (M^T)(v) for power method
        updated_prob_vec = np.dot(transition_matrix, prob_vec)
  
        # Get amount that the updated_prob_vec has changed
        # from the previous iteration, to see if converged
        matrix_diff = np.linalg.norm(np.subtract(updated_prob_vec, prob_vec))
        prob_vec = updated_prob_vec

    # Return the prob_vec, which is the LexRank scores
    return prob_vec


def select_sentences(sorted_sentences, summary_threshold = 0.5):
    """ 
    Takes a list of sentences sorted by LexRank value (descending)
    and selects the sentences to add to the summary greedily based on LexRank value
    while excluding sentences with cosine similarity >= summary_threshold (default = 0.5)
    to any sentence already in the summary.
    Returns a list of selected sentences.
    """

    max_summary_size = 100

    # array of added sentence objects to compare for cosine similarity
    added_sents = []

    # Track the number of tokens so far in the summary
    summary_size = 0

    for sent_index in range(len(sorted_sentences)):

        # add first ranked sentence
        if sent_index == 0:
            added_sents.append(sorted_sentences[sent_index])
               
            # update summary size
            summary_size += sorted_sentences[sent_index].sent_len

        # look at all other sentences            
        else:

            # check if summary hasn't gone over the limit
            if summary_size + sorted_sentences[sent_index].sent_len <= max_summary_size:
                
                # get cos similarity of current sentence with the sentences already added
                cos_sim = [_cosine_similarity(sorted_sentences[sent_index], added_sent) for added_sent in added_sents]          
                            
                # check if any cos sim is at or above the summary_threshold
                similar = any(cos_similarity >= summary_threshold for cos_similarity in cos_sim)

                # if sentence is not similar to any of the already added sentences, add to list
                if not similar:
                    added_sents.append(sorted_sentences[sent_index])

                    # update summary size
                    summary_size += sorted_sentences[sent_index].sent_len

    # Return the list of chosen sentences
    return added_sents


def select_content(topics_list, d = 0.7, intersent_threshold = 0.15, summary_threshold = 0.5, epsilon = 0.1, include_narrative = False, min_sent_len = 8, bias_formula = 0):
    """
    For each topic, creates summaries of <= 100 words (full sentences only) 
    using a Biased LexRank similarity graph algorithm
    with tf-idf cosine similarity and a bias for query topic.

    Args:
        topic_list: a list of Topic objects (which include Documents and Sentences)
        d: damping factor, amount to prioritize topic bias in Markov Matrix
        intersent_threshold: minimum amount of similarity required to include in Similarity Matrix
        summary_threshold: maximum amount of similarity between sentences in summary
        epsilon: minimum amount of difference between probabilities between rounds of power method
        include_narrative: True if the narrative (in addition to title) should be in the bias
        min_sent_len: minimum number of words in a sentence to be used in the summary
        bias_formula: which formula to use - 0 if cosine sim, 1 if ....??? TODO

    Returns:
        topic_list: the modified topic_list from the input, with a list of selected sentences
        in the topic.summary fields of each topic.

    """

    # A dictionary of topics with a list of the (sentence, date) 
    # which are chosen for the summary (<= 100 words)
    topic_summaries = {}
    
    # For each topic, choose the top LexRanked sentences
    # up to 100 words
    for topic in topics_list:

        topic_id = topic.topic_id
        topic_title = topic.title
        topic_docs_list = topic.document_list

        # Get a list of all the sentence objects in this topic
        # Don't include sentences that are less than 5 words
        total_sentences = [sent for doc in topic.document_list for sent in doc.sentence_list if sent.sent_len >= min_sent_len]

        # Build the inter-sentential cosine similarity matrix
        sim_matrix = _build_sim_matrix(total_sentences, intersent_threshold)

        # Build the topic-sentence bias vec
        bias_vec = _build_bias_vec(total_sentences, topic_title)

        # Build a Markov Matrix using the inter-sentential and bias similarities
        markov_matrix = _build_markov_matrix(sim_matrix, bias_vec, d)

        # Get the Biased LexRank for the sentences using the power method
        lex_rank_vec = _power_method(markov_matrix, epsilon)

        # Add the lex_rank to each sentence.score
        for i in range(len(total_sentences)):
            total_sentences[i].score = lex_rank_vec[i]

        # Sort the sentences by score
        sorted_sentences = sorted(total_sentences, reverse=True)

        # Select which sentences to use
        # and add the list to this topic's summary variable
        topic.summary = select_sentences(sorted_sentences, summary_threshold)

    # Return the list of topics now that the summary has been added to each
    return topics_list


