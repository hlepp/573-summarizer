#!opt/python-3.6/bin/python3
# -*- coding: utf-8 -*-

"""Unit tests for content_selection.py"""

__author__ = "Shannon Ladymon"
__email__ = "sladymon@uw.edu"

import unittest
import math
import numpy as np
import sys
sys.path.append("../src")
import content_selection
from data_input import Topic, Document, Sentence, Token
from data_input import build_pseudo_topic

class TestContentSelection(unittest.TestCase):

	# TODO: Add setUp(), tearDown() methods
	# for what to do before/after each test method

	def setUp(self):

		# Set up values for parameters
		self.d = 0.7
		self.threshold = 0.15
		self.epsilon = 0.1

		# Get a topic from the pseudo_topic.txt file
		self.topic = build_pseudo_topic('pseudo_topic_content_selection.txt')
		self.sent_list = [sent for doc in self.topic.document_list for sent in doc.sentence_list]


	def test_cosine_similarity(self):
		
		# Test that a sentence with itself has sim=1
		sim_same_sentence = content_selection._cosine_similarity(self.sent_list[0], self.sent_list[0])
		self.assertEqual(sim_same_sentence, 1)

		# TODO: What's the best way to do this?
		# Get the example sentence values & expected similiarity
		s0_tfidfs = self.sent_list[0].tf_idf
		s1_tfidfs = self.sent_list[1].tf_idf
		num = (s0_tfidfs['This'] * s1_tfidfs['This']) +  (s0_tfidfs['sentence'] * s1_tfidfs['sentence']) + (s0_tfidfs['.'] * s1_tfidfs['.'])

		s0_squared = [num * num for num in s0_tfidfs.values()]	
		d0 = sum(s0_squared)
		s1_squared = [num * num for num in s1_tfidfs.values()]
		d1 = sum(s1_squared)
		expected_s0_s1 = num / (math.sqrt(d0) * math.sqrt(d1))
	
		# Test that two sentences have correct similarity
		# For the pseudo_topic_content_selection.txt file,
		# sent_list[0] and sent_list[1] should have sim= 0.3407
		sim_s0_s1 = content_selection._cosine_similarity(self.sent_list[0], self.sent_list[1])
		self.assertAlmostEqual(sim_s0_s1, expected_s0_s1, places=4)


	def test_build_sim_matrix(self):

		# When no sentences, returns an empty np matrix (size==0)
		sim_matrix_none = content_selection._build_sim_matrix([], self.threshold)
		self.assertEqual(sim_matrix_none.size, 0)

		# When 1 sentence, returns an np matrix with one value
		sim_matrix_one = content_selection._build_sim_matrix([self.sent_list[0]], self.threshold)
		self.assertEqual(sim_matrix_one[0][0], 1.0)

		# When 2 different sentences, return a 2x2 matrix with sims
		sim_matrix_two = content_selection._build_sim_matrix([self.sent_list[0], self.sent_list[1]], self.threshold)
		sim_s0_s1 = content_selection._cosine_similarity(self.sent_list[0], self.sent_list[1])
		row_sum = sim_s0_s1 + 1.0

		self.assertAlmostEqual(sim_matrix_two[0][0], 1.0/row_sum)
		self.assertAlmostEqual(sim_matrix_two[0][1], sim_s0_s1/row_sum) 
		self.assertAlmostEqual(sim_matrix_two[1][0], sim_s0_s1/row_sum) 
		self.assertAlmostEqual(sim_matrix_two[1][1], 1.0/row_sum) 


	def test_build_bias_vec(self):

		# When no sentences, returns an empty vector (size==0)
		bias_vec_none = content_selection._build_bias_vec([], self.topic.title)
		self.assertEqual(bias_vec_none.size, 0)

		# When one sentence which is identical to the topic
		bias_vec_identical = content_selection._build_bias_vec([self.topic.title], self.topic.title)
		expected_identical_sim = content_selection._cosine_similarity(self.topic.title, self.topic.title)
#		print("TESTING: identical_sim={}\ntopic type={}".format(expected_identical_sim, type(self.topic.title)))
#		print("TESTING: title.token_list={}, title.tf_idf={}".format(self.topic.title.token_list, self.topic.title.tf_idf))
#		print("TESTING identical={}".format(bias_vec_identical[0]))

		# TODO: currently this won't work since topic.title doesn't have tokens or tf_idf
		# wait until that is fixed, and then this should work
#		self.assertEqual(bias_vec_identical[0], 1.0)

		# When one sentence which is different from the topic
		bias_vec_diff = content_selection._build_bias_vec([self.sent_list[0]], self.topic.title)
		expected_diff_sim = content_selection._cosine_similarity(self.sent_list[0], self.topic.title)
#		self.assertAlmostEqual(bias_vec_diff[0], expected_diff_sim)

	
	def test_build_markov_matrix(self):

		# When empty sim_matrix and bias_vec, returns an empty matrix
		sim_matrix_none = np.zeros((0,0))
		bias_vec_none = np.zeros(0)
		markov_matrix_none = content_selection._build_markov_matrix(sim_matrix_none, bias_vec_none, self.d)
		self.assertEqual(markov_matrix_none.size, 0)

		# When sim_matrix has one element of 1 and bias_vec has 0
		expected_m_m_one = 1 - self.d
		sim_matrix_one = np.array([1.0])	
		bias_vec_one_empty = np.zeros(1)
		markov_matrix_one = content_selection._build_markov_matrix(sim_matrix_one, bias_vec_one_empty, self.d)
		self.assertAlmostEqual(markov_matrix_one[0], expected_m_m_one)

		# When sim_matrix has one element of 1 and bias_vec has 1
		expected_m_m_one_one = 1.0
		bias_vec_one = np.array([1.0])
		markov_matrix_one_one = content_selection._build_markov_matrix(sim_matrix_one, bias_vec_one, self.d)
		self.assertAlmostEqual(markov_matrix_one_one[0], expected_m_m_one_one)


	def test_power_method(self):
		pass
		# TODO: What should be tested here?


	def test_select_sentences(self):
		pass
		# TODO: test somehow (maybe change to add a second doc with a different date)


	def test_select_content(self):
		pass
		# TODO: figure out what to test on


if __name__ == '__main__':
	unittest.main()
