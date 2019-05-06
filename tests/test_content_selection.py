#!opt/python-3.6/bin/python3
# -*- coding: utf-8 -*-

"""Unit tests for content_selection.py"""

__author__ = "Shannon Ladymon"
__email__ = "sladymon@uw.edu"

import unittest
import sys
sys.path.append("../src")
from content_selection import select_content
from data_input import Topic, Document, Sentence, Token
#import data_input

class TestContentSelection(unittest.TestCase):

	# TODO: Add setUp(), tearDown() methods
	# for what to do before/after each test method

	def setUp(self):

		# Set up values for parameters
		self.d = 0.7
		self.threshold = 0.15
		self.epsilon = 0.1

		# Set up the topics
		self.topic1 = Topic("Topic1", "A", "Testing topic 1")
		self.topic2 = Topic("Topic2", "A", "Testing topic 2")

		# Set up the documents
		self.doc1 = Document(self.topic1, "doc1")
		self.doc2 = Document(self.topic2, "doc2")
		self.topic1.document_list = [self.doc1]
		self.topic2.document_list = [self.doc2]

		# Set up the sentences
		self.sent1a = Sentence(self.doc1, "This sentence is testing topic 1 for content selection.")
		self.sent1b = Sentence(self.doc1, "This sentence is also testing topic 1 for content selection.")
		self.sent2a = Sentence(self.doc2, "This sentence is testing topic 2.")
		self.sent2b = Sentence(self.doc2, " ")
		self.doc1.sentence_list = [self.sent1a, self.sent1b]
		self.doc2.sentence_list = [self.sent2a, self.sent2b]

		# TODO: get tokenization done


		# TODO: set up idf for each topic, and tf_idf for each sentence

		self.topics_list = [self.topic1, self.topic2]
		# TODO: create list of topics w/documents & sentences

	def _cosine_similarity(self):
		pass

	def _build_sim_matrix(self):

		print("TESTING: in unit test for _build_sim_matrix")
		# TODO: think through each edge case

		# When no sentences?

		# When 1 sentence

		# When 2 sentences

	def test_select_content(self):
		pass
		# TODO: fix to work with select_content
#		topic_summaries = select_content(1)

#		self.assertEqual(len(summaries_sent),2)
#		self.assertEqual(len(sent_dates), 2)


if __name__ == '__main__':
	unittest.main()
