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

            # If the same sentence, default sim is 1.0
            sim = 1.0

            # For any different sentences
            if i != j:
                sim = _cosine_similarity(sent_list[i], sent_list[j])

                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim

    # Normalize by dividing by sum of each row
    row_sums = sim_matrix.sum(axis=1, keepdims=True)

    # Change any row sum that is 0 to 1 to avoid division by 0
    if 0 in row_sums:
        row_sums[row_sums == 0] = 1


    sim_matrix = sim_matrix / row_sums

    return sim_matrix


def _build_bias_vec(sent_list, topic_sent):
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
        bias_vec[i] = _cosine_similarity(sent_list[i], topic_sent)

    # Normalize by dividing by the sum
    bias_sum = np.sum(bias_vec)

    # If the bias sum is 0, change to 1 to avoid division by 0
    if bias_sum == 0:
        bias_sum = 1

    bias_vec = bias_vec / bias_sum

    return bias_vec


def _build_markov_matrix(sim_matrix, bias_vec, d):
    """
    Builds and returns the markov matrix (matrix to multiply by in the power method)
    using the Biased LexRank formula with cosine similarity.
    """
    markov_matrix = (d * (bias_vec)) + ((1-d) * (sim_matrix))
    # TODO: what to do for the Biased LexRank continuous part (multiply by the probs at the end?)

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

def select_content(topics_list):
    """
    TODO: Update once we've figured out what input this function receives
    """
    # Default hyperparameter values 
    # TODO: do tuning later to choose better values
    d = 0.7  # damping factor, prioritizes similiarity with topic over inter-senential
    threshold = 0.15  # inter-sentential similarity must be above
    epsilon = 0.1  # in power method, matrix diff between iterations must be above


    # TODO: Possibly do a check that topics_list is a list of Topic objects?

    # A dictionary of topics with a list of the (sentence, date) 
    # which are chosen for the summary (<= 100 words)
    topic_summaries = {}
    


    # For each topic, choose the top LexRanked sentences
    # up to 100 words
    for topic in topics_list:

        topic_id = topic.topic_id
        topic_title = topic.title
        topic_docs_list = topic.document_list
        # TODO: Possibly grab the other variables like narrative

        # List to hold tuples of (sentence, date)
        # for each sentence chosen for the summary
        chosen_sent = []

        # Track the number of tokens so far in the summary
        summary_size = 0

        # Get a list of all the sentence objects in this topic
        # Don't include sentences that are less than 5 words
        total_sentences = [sent for doc in topic.document_list for sent in doc.sentence_list if sent.sent_len >=5]


        # Build the inter-sentential cosine similarity matrix
        sim_matrix = _build_sim_matrix(total_sentences, threshold)


        bias_vec = _build_bias_vec(total_sentences, topic_title)


        markov_matrix = _build_markov_matrix(sim_matrix, bias_vec, d)

        lex_rank_vec = _power_method(markov_matrix, epsilon)


        # Add the lex_rank to each sentence.score
        for i in range(len(total_sentences)):
            total_sentences[i].score = lex_rank_vec[i]

        # Sort the sentences by score
        sorted_sentences = sorted(total_sentences, reverse=True)

        # Sentence selection- add sentences greedily based on ranking
        # Sentences with high similarity to added sentences are not added
                
        # array of added sentence objects to compare for cosine similarity
        added_sents = []

        for sent_index in range(len(sorted_sentences)):

            # add first ranked sentence
            if sent_index == 0:
                chosen_sent.append((sorted_sentences[sent_index].original_sentence, sorted_sentences[sent_index].parent_doc.date))
                added_sents.append(sorted_sentences[sent_index])
               
                # update summary size
                summary_size += sorted_sentences[sent_index].sent_len

            # look at all other sentences            
            else:

                # check if summary hasn't gone over the limit
                if summary_size + sorted_sentences[sent_index].sent_len <= 100:
                
                    # get cos similarity of current sentence with the sentences already added
                    cos_sim = [_cosine_similarity(sorted_sentences[sent_index], added_sent) for added_sent in added_sents]          
                            
                    # check if any cos sim is at or above threshold= 0.5
                    similar = any(cos_similarity >= 0.5 for cos_similarity in cos_sim)

                    # if sentence is not similar to any of the already added sentences, add to chosen
                    if not similar:
                        chosen_sent.append((sorted_sentences[sent_index].original_sentence, sorted_sentences[sent_index].parent_doc.date))
                        added_sents.append(sorted_sentences[sent_index])

                        # update summary size
                        summary_size += sorted_sentences[sent_index].sent_len


        # Add the chosen_sent to the topic_summaries dict
        topic_summaries[topic_id] = chosen_sent

    # Return a dictionary of {topic: [(sent, date)]}
    return topic_summaries


