#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Benny Longwill"
__email__ = "longwill@uw.edu"

from lxml import etree
from lxml import html

'''#####################################
-Class Object used to more easily retrieve raw document data from Acquaint and Acquaint2 Databases.
-Configures document ID to create document file path and to determine database
-Determines HTML or XML parsing and Tags used for internal element searching
-Caches XML/HTML files as they are parsed in Dictionary with path-file format 
'''#######################################

class Document_Retriever:
    def __init__(self):
        self.xml_parser_cache = {}
        self.date=None
        self.headline_tag = None
        self.category_tag = None
        self.dateline_tag = None
        self.text_tag = None

    #Contains lots of the hardcoded configurations in order to parse doc id to derive the correct database
    def configure(self, doc_id: str):
        self.doc_id = doc_id

        if "_" in doc_id:
            source, lang, other = doc_id.split("_")
            self.date, specifier = other.split(".")
            year = self.date[:4]
        else:
            source = doc_id[:3]
            self.date = doc_id[3:11]
            year = self.date[:4]
            # specifier = doc_id[-4:]
            lang = "ENG"

        if source == "XIE":
            alt_source = "XIN"
        else:
            alt_source = source

        #############################
        self.acquaint = int(year) >= 1996 and int(year) <= 2000

        self.acquaint2 = int(year) >= 2004 and int(year) <= 2006

        if self.acquaint:

            if alt_source != 'NYT':
                alt_source = alt_source + "_ENG"

            self.headline_tag = 'headline'
            self.category_tag = 'category'
            self.dateline_tag = 'date_time'
            self.text_tag = 'text'

            self.doc_path = "/corpora/LDC/LDC02T31/" + source.lower() + "/" + year + "/" + date + "_" + alt_source.upper()

        elif self.acquaint2:

            self.headline_tag = 'HEADLINE'
            self.dateline_tag = 'DATELINE'
            self.text_tag = 'TEXT'
            self.category_tag = None

            self.doc_path = "/corpora/LDC/LDC08T25/data/" + alt_source.lower() + "_" + lang.lower() + "/" + alt_source.lower() + "_" + lang.lower() + "_" + date[
                                                                                                                                                            :6] + ".xml"
    #Method requires document ID to determine database and document file path.
    # Retrieves raw document.
    def retrieve_doc(self, doc_id):
        self.configure(doc_id)
        raw_doc = None
        if self.acquaint:
            if self.doc_path in self.xml_parser_cache:
                tree = self.xml_parser_cache[self.doc_path]
            else:
                parser = etree.HTMLParser(encoding='utf-8', remove_blank_text=True)
                with open(self.doc_path) as file:
                    tree = html.fragment_fromstring(file.read(), create_parent='body', parser=parser)
                self.xml_parser_cache.update({self.doc_path: tree})

            raw_doc = [element for element in tree.findall("doc") if element.find("docno").text == " " + doc_id + " "][
                0]

        elif self.acquaint2:
            if self.doc_path in self.xml_parser_cache:
                tree = self.xml_parser_cache[self.doc_path]
            else:
                tree = etree.parse(self.doc_path)
                self.xml_parser_cache.update({self.doc_path: tree})

            raw_doc = [element for element in tree.findall("DOC") if element.get("id") == doc_id][0]

        return raw_doc


