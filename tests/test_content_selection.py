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
from data_input import build_pesudo_topic
#sys.path.remove("../src")

class TestContentSelection(unittest.TestCase):

	# TODO: Add setUp(), tearDown() methods
	# for what to do before/after each test method

	def setUp(self):

		# Set up values for parameters
		self.d = 0.7
		self.threshold = 0.15
		self.epsilon = 0.1

		# Get a topic from the pseudo_topic.txt file
		topic = build_pesudo_topic('../tests/pseudo_topic.txt')
		print("TESTING: topic={}".format(topic.title))


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
