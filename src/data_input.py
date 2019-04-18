#!/usr/bin/env python3

from nltk import sent_tokenize
from nltk import word_tokenize
from bs4 import BeautifulSoup
import document_retriever

#### Put none for non-existing text
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
### Tag Variables used to avoid hardcoding later in the document or in case of tag changes

title_tag='title'
narrative_tag='narrative'
docsetA_tag='docseta'
id_tag='id'
doc_tag='doc'
topic_tag='topic'
parser_tag='lxml'

###############################

# Takes a list of file paths, collects all data and stores into a list of class object 'Topic' data structures
def get_data(file_paths:list)->list:

    task_data = ""
    for path in file_paths:
        task_data += open(path).read() + "\n"

    return get_topics_list(task_data)

#Takes Document object and the text from doc file. The block of text is separated into sentences as sentence objects and also tokenized using NLTK.
def populate_sentence_list(current_doc, doc_text):

    doc_sentences = sent_tokenize(doc_text)

    for doc_sentence in doc_sentences:
        current_doc.sentence_list.append(Sentence(word_tokenize(doc_sentence)))  ############## Creates sentence object
        current_doc.sent_count += 1
#Takes a Topic class object, an xml or html document set element, and a document retriever object
# Itterates all document Id's in html/xml element and uses the doc retriever to get the raw document from database
#Extracts attributes from raw document and creates Document class object, fills it with sentence objects and adds it to the current topic object
def populate_document_list(current_topic, docsetA, doc_ret:document_retriever.Document_Retriever):

    for doc in docsetA.findAll(doc_tag):

        doc_id = doc.attrs[id_tag]

        raw_doc = doc_ret.retrieve_doc(doc_id)
        headline, category, dateline, doc_text = get_doc_attributes(raw_doc, doc_ret.headline_tag, doc_ret.category_tag, doc_ret.dateline_tag, doc_ret.text_tag)

        current_doc = Document(doc_id=doc_id, headline=headline, dateline=dateline, category=category)  ########## Creates document object

        populate_sentence_list(current_doc, doc_text)

        current_topic.document_list.append(current_doc)
# Takes the raw topic xml/html and a set of title, narrative, and docset TAGS according to the format of file
# extracts the respective text including document IDs for Aqcuaint(2) database
def get_topic_attributes(raw_topic, title_tag:str , narrative_tag:str, docsetA_tag:str):

    narrative=None
    title=None

    topic_id = raw_topic.attrs[id_tag]

    if title_tag:
        title_element = raw_topic.find(title_tag)
        if title_element is not None:
            title = title_element.text

    if narrative_tag:
        narrative_element = raw_topic.find(narrative_tag)
        if narrative_element is not None:
            narrative = narrative_element.text

    docsetA = raw_topic.find(docsetA_tag)

    docsetA_id = docsetA.attrs[id_tag]

    return topic_id, title, narrative, docsetA_id, docsetA

# Requires raw document and the specific tags used according to current HTML or XML.
# Extracts the headline category dateline and text if available in raw document.
def get_doc_attributes(raw_doc, headline_tag:str="", category_tag:str="", dateline_tag:str="", text_tag:str=""):

    ### Instantiate variables in case text not available in raw doc
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

#Parses Raw data XML file argument and extracts the topic elements.
# Creates Topic elements and fills withtopic attributes and Document objects
# Returns these topic objects as a list.
def get_topics_list(task_data:str)->list:
    topics=[]
    doc_ret = document_retriever.Document_Retriever()

    soup = BeautifulSoup(task_data, parser_tag)
    raw_topics = soup.findAll(topic_tag)

    for raw_topic in raw_topics:
        topic_id, title, narrative, docsetA_id, docsetA = get_topic_attributes(raw_topic, title_tag, narrative_tag, docsetA_tag)

        current_topic= Topic(topic_id=topic_id,title=title,narrative=narrative,docsetA_id=docsetA_id) ########### Creates topic object

        populate_document_list(current_topic, docsetA, doc_ret)

        topics.append(current_topic)

    return topics

###############################