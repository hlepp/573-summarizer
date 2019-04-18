#!/usr/bin/env python3

import nltk
from lxml import etree
from lxml import html
from bs4 import BeautifulSoup

#### Put none for non-existing
###############################
class Topic:
    def __init__(self, topic_id:str=None ,docsetA_id:str=None,  title:str=None , narrative:str=None):
        self.topic_id=topic_id
        self.title=title
        self.narrative=narrative ##### *** Not all topics have this attribute ***
        self.docsetA_id=docsetA_id
        self.document_list=[]
        self.summary = []
class Document:
    def __init__(self, doc_id:str=None, headline:str=None,dateline:str=None, category:str=None):
        self.doc_id=doc_id
        self.headline=headline ##### *** Not all topics have this attribute ***
        self.dateline=dateline
        self.category=category ##### *** Not all topics have this attribute ***
        self.sent_count=0
        self.sentence_list=[]
class Sentence:
    def __init__(self, tokens:list=None):
        self.score=0
        self.tokens=tokens

###############################

def get_data(file_paths:list)->list:

    task_data = ""
    for path in file_paths:
        task_data += open(path).read() + "\n"

    return get_topics_list(task_data)

def process_doc_id(doc_id:str):
    if "_" in doc_id:
        source, lang, other = doc_id.split("_")
        date, specifier = other.split(".")
        year = date[:4]
    else:
        source = doc_id[:3]
        date = doc_id[3:11]
        year = date[:4]
        #specifier = doc_id[-4:]
        lang = "ENG"

    if source == "XIE":
        alt_source = "XIN"
    else:
        alt_source = source

    return source, lang, date, year, alt_source

def populate_sentence_list(current_doc, doc_text):

    doc_sentences = nltk.sent_tokenize(doc_text)

    for doc_sentence in doc_sentences:
        current_doc.sentence_list.append(Sentence(nltk.word_tokenize(doc_sentence)))  ############## Creates sentence object
        current_doc.sent_count += 1

def populate_document_list(current_topic, docsetA, xml_parser_cache):

    for doc in docsetA.findAll("doc"):

        category = None
        headline = None
        dateline = None
        doc_text = None

        doc_id = doc.attrs['id']
        source, lang, date, year, alt_source = process_doc_id(doc_id)

        if int(year) >= 1996 and int(year) <= 2000:

            headline, category, dateline, doc_text = get_acquaint_doc_attributes(doc_id, source, year, date, alt_source,
                                                                                 xml_parser_cache)

        elif int(year) >= 2004 and int(year) <= 2006:

            headline, category, dateline, doc_text = get_acquaint2_doc_attributes(doc_id, lang, date, alt_source,
                                                                                  xml_parser_cache)

        current_doc = Document(doc_id=doc_id, headline=headline, dateline=dateline,
                               category=category)  ########## Creates document object

        populate_sentence_list(current_doc, doc_text)

        current_topic.document_list.append(current_doc)


def get_topic_attributes(raw_topic, title_tag:str , narrative_tag:str, docsetA_tag:str):

    narrative=None
    title=None

    topic_id = raw_topic.attrs['id']

    if title_tag:
        title_element = raw_topic.find(title_tag)
        if title_element is not None:
            title = title_element.text

    if narrative_tag:
        narrative_element = raw_topic.find(narrative_tag)
        if narrative_element is not None:
            narrative = narrative_element.text

    docsetA = raw_topic.find(docsetA_tag)

    docsetA_id = docsetA.attrs['id']

    return topic_id, title, narrative, docsetA_id, docsetA

def get_acquaint_doc_attributes(doc_id, source, year, date, alt_source,xml_parser_cache):
    if alt_source != 'NYT':
        alt_source = alt_source + "_ENG"
    doc_path = "/corpora/LDC/LDC02T31/" + source.lower() + "/" + year + "/" + date + "_" + alt_source.upper()

    if doc_path in xml_parser_cache:
        tree = xml_parser_cache[doc_path]
    else:
        parser = etree.HTMLParser(encoding='utf-8', remove_blank_text=True)
        tree = html.fragment_fromstring(open(doc_path).read(), create_parent='body', parser=parser)
        xml_parser_cache.update({doc_path: tree})

    raw_doc = [element for element in tree.findall("doc") if element.find("docno").text == " " + doc_id + " "][0]


    return get_doc_attributes(raw_doc, 'headline', 'category', 'date_time','text')

def get_acquaint2_doc_attributes(doc_id, lang, date, alt_source,xml_parser_cache):
    doc_path = "/corpora/LDC/LDC08T25/data/" + alt_source.lower() + "_" + lang.lower() + "/" + alt_source.lower() + "_" + lang.lower() + "_" + date[:6] + ".xml"
    if doc_path in xml_parser_cache:
        tree = xml_parser_cache[doc_path]
    else:
        tree = etree.parse(doc_path)
        xml_parser_cache.update({doc_path: tree})

    raw_doc = [element for element in tree.findall("DOC") if element.get("id") == doc_id ][0]


    return get_doc_attributes(raw_doc=raw_doc, headline_tag='HEADLINE', dateline_tag='DATELINE', text_tag='TEXT')


def get_doc_attributes(raw_doc, headline_tag:str="", category_tag:str="", dateline_tag:str="", text_tag:str=""):

    headline = None
    category = None
    dateline = None
    doc_text = None

    if headline_tag:
        headline_element = raw_doc.find(headline_tag)
        if headline_element is not None:
            headline = headline_element.text

    if category_tag:
        category_element = raw_doc.find(category_tag)
        if category_element is not None:
            category = category_element.text

    if dateline_tag:
        dateline_element = raw_doc.find(dateline_tag)
        if dateline_element is not None:
            dateline = dateline_element.text

    if text_tag:
        doc_text=' '.join(raw_doc.find(text_tag).itertext())


    return headline, category, dateline, doc_text



def get_topics_list(task_data:str)->list:
    topics=[]
    xml_parser_cache = {}

    soup = BeautifulSoup(task_data, "lxml")
    raw_topics = soup.findAll('topic')

    for raw_topic in raw_topics:
        topic_id, title, narrative, docsetA_id, docsetA = get_topic_attributes(raw_topic=raw_topic, title_tag='title', narrative_tag='narrative', docsetA_tag="docseta")

        current_topic= Topic(topic_id=topic_id,title=title,narrative=narrative,docsetA_id=docsetA_id) ########### Creates topic object

        populate_document_list(current_topic, docsetA ,xml_parser_cache)

        topics.append(current_topic)

    return topics

###############################




























