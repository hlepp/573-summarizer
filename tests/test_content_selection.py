#!opt/python-3.6/bin/python3
# -*- coding: utf-8 -*-

"""Unit tests for content_selection.py"""

__author__ = "Shannon Ladymon"
__email__ = "sladymon@uw.edu"

import unittest
import sys
sys.path.append("../src")
from content_selection import select_content

class TestContentSelection(unittest.TestCase):

	# TODO: Add setUp(), tearDown() methods
	# for what to do before/after each test method

	def test_select_content(self):
		# Dummy test temporarily
		# TODO: fix to work with select_content

		summaries_sent, sent_dates = select_content(1)

		self.assertEqual(len(summaries_sent),2)
		self.assertEqual(len(sent_dates), 2)


if __name__ == '__main__':
	unittest.main()
