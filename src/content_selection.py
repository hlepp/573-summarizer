#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Content Selector for multi-document text summarization that ranks and chooses the sentences to use in the summary."""

__author__ = "Shannon Ladymon"
__email__ = "sladymon@uw.edu"

from data_input import Topic, Document, Sentence

def select_content(topics_list):
	"""
	TODO: Update once we've figured out what input this function receives
	"""
	# TODO: Possibly do a check that topics_list is a list of Topic objects?

	# A dictionary of topics with a list of the (sentence, date) 
	# which are chosen for the summary (<= 100 words)
	topic_summaries = {}
	

	# TODO: Use an actual algorithm later
	# For now, adds sentences while <= 100 tokens
	# TODO: check if number of tokens should include punctuation, etc.
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

		# Add sentences as long as there are <= 100 tokens
		for doc in topic_docs_list:

			doc_date = doc.dateline

			for sent_obj in doc.sentence_list:

				sent = sent_obj.original_sentence

				# Sentence length is the number of words
				# which are space delimited
				sent_len = sent.count(" ") + 1

				# Check if this sentence can be added to the summary
				# (if there is room) and do so if possible
				if summary_size + sent_len <= 100:
					summary_size += sent_len
					chosen_sent.append((sent, doc_date))

		# Add the chosen_sent to the topic_summaries dict
		topic_summaries[topic_id] = chosen_sent

	# Return a dictionary of {topic: [(sent, date)]}
	return topic_summaries


