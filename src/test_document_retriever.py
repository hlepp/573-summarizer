#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Benny Longwill"
__email__ = "longwill@uw.edu"

import unittest
import document_retriever



class TestDocumentRetriever(unittest.TestCase):
    def test_retrieve_doc(self):
        doc_ret=document_retriever.Document_Retriever()

        raw_doc = doc_ret.retrieve_doc("APW19990421.0284")
        self.assertEqual(doc_ret.date, str(19990421))
        self.assertEqual(doc_ret.acquaint, 1999 >= 1996 and 1999 <= 2000)
        self.assertEqual(doc_ret.acquaint2, 1999 >= 2004 and 1999 <= 2006)
        self.assertEqual(doc_ret.doc_path, '/corpora/LDC/LDC02T31/apw/1999/19990421_APW_ENG')
        self.assertEqual(doc_ret.headline_tag, 'headline')
        self.assertEqual(doc_ret.category_tag, 'category')
        self.assertEqual(doc_ret.dateline_tag, 'date_time')
        self.assertEqual(doc_ret.text_tag, 'text')


        raw_doc=doc_ret.retrieve_doc("XIN_ENG_20041113.0001")
        self.assertEqual(doc_ret.date, str(20041113))
        self.assertEqual(doc_ret.acquaint, 2004 >= 1996 and 2004 <= 2000)
        self.assertEqual(doc_ret.acquaint2, 2004 >= 2004 and 2004 <= 2006)
        self.assertEqual(doc_ret.doc_path, '/corpora/LDC/LDC08T25/data/xin_eng/xin_eng_200411.xml' )
        self.assertEqual(doc_ret.headline_tag, 'HEADLINE')
        self.assertEqual(doc_ret.category_tag, None)
        self.assertEqual(doc_ret.dateline_tag, 'DATELINE')
        self.assertEqual(doc_ret.text_tag, 'TEXT')






if __name__ == '__main__':
    unittest.main()
