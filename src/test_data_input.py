#!/usr/bin/env python3
#/opt/python-3.6/bin/python3 #### Needed if using blingfire library

# -*- coding: utf-8 -*-

__author__ = "Benny Longwill"
__email__ = "longwill@uw.edu"

import unittest
import data_input

file_path = '/dropbox/18-19/573/Data/Documents/training/2009/UpdateSumm09_test_topics.xml'
#'/dropbox/18-19/573/Data/Documents/devtest/GuidedSumm10_test_topics.xml'

#'/dropbox/18-19/573/Data/Documents/training/2009/UpdateSumm09_test_topics.xml'
                    # , # '/dropbox/18-19/573/Data/Documents/evaltest/GuidedSumm11_test_topics.xml']  ####### Shouldn't be used until deliverable 4


class TestDataInput(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.topics = data_input.get_data(file_path)
        cls.topic = [topic for topic in cls.topics if topic.topic_id == "D0901A"][0]
        cls.document= [document for document in cls.topic.document_list if document.doc_id == "XIN_ENG_20041113.0001"][0]
        cls.sentence = cls.document.sentence_list[0]

    def test_get_topics_list(self):
        ## Check that get_input actually returns a list of Topic objects
        self.assertGreater(len(self.topics), 0)
        self.assertEqual(type(self.topics), type(list()))
        self.assertEqual(type(self.topic) , type(data_input.Topic()))

    def test_get_topic_attributes(self):
        ### Checks topic to see if all attributes were correctly gathered
        self.assertEqual(self.topic.title.original_sentence, "Indian Pakistan conflict" )
        self.assertEqual(self.topic.narrative.original_sentence, "Describe efforts made toward peace in the India-Pakistan conflict over Kashmir.")
        self.assertEqual(self.topic.docsetA_id, "D0901A-A" )
    def test_populate_document_list(self):
        self.assertGreater(len(self.topic.document_list), 0)
        self.assertEqual(self.topic.document_list[0].doc_id, "XIN_ENG_20041113.0001")

    def test_get_doc_attributes(self, cls):
        cls.topic = [topic for topic in self.topics if topic.topic_id == "D1001A"][0]
        cls.document = [document for document in self.topic.document_list if document.doc_id == "APW19990421.0284"][0]
        self.assertEqual(self.document.headline.original_sentence, "Explosive Devices Slow Body Count")
        self.assertEqual(self.document.category, " usa ")

    def test_populate_sentence_list(self):
        self.assertEqual(self.sentence.parent_doc, self.document)
        self.assertGreater(self.document.sent_count, 0)
        self.assertGreater(len(self.document.sentence_list), 0)
        self.assertIn("LITTLETON, Colo. (AP) -- The sheriff's", self.sentence.original_sentence)
        self.assertIn("They said their priority was making sure the school was safe", self.document.sentence_list[self.document.sent_count-1].original_sentence)

    def test_populate_token_list(self):
        self.assertGreater(len(self.sentence.token_list), 0)


if __name__ == '__main__':
    unittest.main()
