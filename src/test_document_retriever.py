#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Benny Longwill"
__email__ = "longwill@uw.edu"

import unittest
import document_retriever



class TestDocumentRetriever(unittest.TestCase):
    def test_retrieve_doc(self):
        doc_ret=document_retriever.Document_Retriever()
        raw_doc=doc_ret.retrieve_doc("XIN_ENG_20041113.0001")

        print(raw_doc)

        raw_doc = doc_ret.retrieve_doc("APW19990421.0284")


if __name__ == '__main__':
    unittest.main()
