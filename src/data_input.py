#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Benny Longwill"
__email__ = "longwill@uw.edu"

from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

from bs4 import BeautifulSoup
import document_retriever
from math import log
#from blingfire import text_to_words, text_to_sentences

stop_words = set(stopwords.words('english'))


#### Put none for non-existing text
###############################
class Topic:
    def __init__(self, topic_id:str=None ,docsetA_id:str=None,  title:str=None , narrative:str=None):
        self.topic_id=topic_id
        self.title=title
        self.narrative=narrative ##### *** Not all topics have this attribute ***
        self.docsetA_id=docsetA_id
        self.doc_count=0
        self.sent_count=0
        self.document_list=[]
        self.summary = []
        self.idf={}

    #Returns list of all sentences in Topic, including Title sentence, narrative sentences, and headlines
    def all_sentences(self)->list:

        total_sentences=[sentence for document in self.document_list for sentence in document.sentence_list]

        # checks the title in the topic
        if self.title:
            total_sentences.append(self.title)
        # checks the narrative in the topic
        if self.narrative:
            total_sentences.append(self.narrative)

            # checks all headlines if available in every document in the topic
        total_sentences += [document.headline for document in self.document_list if document.headline]

        return total_sentences

    ##### Counts the sentences under this topic that contains a token parameter
    def n_containing(self, token):
        # checks all tokens in each sentence of every document in the topic
        count = sum(1 for document in self.document_list for sentence in document.sentence_list if
            token in sentence)

        # checks all headlines if available in every document in the topic
        count += sum(1 for document in self.document_list if document.headline if
                    token in document.headline)

        # checks the title in the topic
        if self.title and token in self.title:
            count+=1
        # checks the narrative in the topic
        if self.narrative and token in self.narrative:
            count+=1

        return count


    #Take the ratio of the total number of documents to the number of documents containing any word.
    #Then it adds 1 to the devisor to avoid zero division and then takes the log otherwise the weight will be too high
    def get_idf(self, token):
        if token in self.idf:
           return self.idf[token]
        else:
            current_idf = log(self.sent_count / (1 + self.n_containing(token)))
            self.idf[token]=current_idf
            return current_idf

    # Must be used after all Documents, Sentences, and Tokens have been filled.
    def compute_tf_idf(self):
        for doc in self.document_list:

            for sentence in doc.sentence_list + [self.title, self.narrative, doc.headline ]:
                if sentence:

                    for token in sentence.token_list:
                        token_value=token.token_value

                        sentence.tf_idf[token]=token.tf*self.get_idf(token_value)

class Document:
    def __init__(self, parent_topic:Topic , doc_id:str, headline:str=None,date:str=None, category:str=None):
        self.parent_topic=parent_topic
        self.doc_id=doc_id
        self.headline=headline ##### *** Not all topics have this attribute ***
        self.date=date
        self.category=category ##### *** Not all topics have this attribute ***
        self.sent_count=0
        self.sentence_list=[]
        self.parent_topic.doc_count+=1 ### Increments parent count When initialized

    def __repr__(self):
        return " ".join(self.sentence_list)
    # Less than method allows for sorting
    def __lt__(self, other):
        return (self.date < other.date)

class Sentence:
    def __init__(self,parent_doc:Document or Topic, original_sentence:str):
        self.score=0
        self.parent_doc=parent_doc
        self.original_sentence=original_sentence
        self.sent_len = original_sentence.count(" ") + 1   #Counts words in original sentence
        self.token_list=[]   ##### *** List of non-duplicate Tokens as Objects ***
        self.tf_idf={}

        # Increments parent count when initialized. If parent_doc is Document, then Topic sent_count is also incremented
        self.parent_doc.sent_count+=1
        if type(self.parent_doc) is Document:
            self.parent_doc.parent_topic.sent_count+=1

    def __repr__(self):
        return self.original_sentence
    # Less than method allows for sorting
    def __lt__(self, other):
        return (self.score < other.score)
    def __eq__(self, other):
        return (self.score == other.score)
    # allows the use of 'is' operator on Sentence object
    def __contains__(self, param):
        return any(token.token_value == param for token in self.token_list)

class Token:
    def __init__(self,parent_sentence:Sentence, token_value:str, tf):
        self.tf=tf
        self.parent_sentence=parent_sentence
        self.token_value=token_value
    def __repr__(self):
        return self.token_value


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
        with open(path) as file:
            task_data += file.read() + "\n"

    return get_topics_list(task_data)

#Tokenizes a sentences and populates the sentence object with a token object list
def populate_token_list(current_sent:Sentence, original_sentence:str):

    #Edits the original sentence to remove article formatting
    # This location was chosen so that all sentences that are tokenized (i.e., Title sentence) could also be effected by this
    original_sentence = original_sentence.replace("\n", " ").strip().replace("  ", " ")
    current_sent.original_sentence=original_sentence

    tokenized_sent = word_tokenize(original_sentence)
    tokenized_sent_len = len(tokenized_sent)
    for token in set(tokenized_sent):
        ''''######### Removes stop words #########'''''
        if token not in stop_words:
            current_sent.token_list.append(Token(current_sent, token, tokenized_sent.count(token) / tokenized_sent_len))

#Takes Document object and the text from doc file. The block of text is separated into sentences as sentence objects and also tokenized using NLTK.
def populate_sentence_list(current_doc, doc_text):

    doc_sentences = sent_tokenize(doc_text)

    for doc_sentence in doc_sentences:

        current_sentence=Sentence(current_doc, doc_sentence)

        populate_token_list(current_sentence, doc_sentence)

        current_doc.sentence_list.append( current_sentence )  ############## Creates sentence object

#Takes a Topic class object, an xml or html document set element, and a document retriever object
# Itterates all document Id's in html/xml element and uses the doc retriever to get the raw document from database
#Extracts attributes from raw document and creates Document class object, fills it with sentence objects and adds it to the current topic object
def populate_document_list(current_topic, docsetA, doc_ret:document_retriever.Document_Retriever):

    for doc in docsetA.findAll(doc_tag):

        doc_id = doc.attrs[id_tag]

        raw_doc = doc_ret.retrieve_doc(doc_id)
        headline, category, dateline, doc_text = get_doc_attributes(raw_doc, doc_ret.headline_tag, doc_ret.category_tag, doc_ret.dateline_tag, doc_ret.text_tag)

        current_doc = Document(parent_topic=current_topic, doc_id=doc_id, date=doc_ret.date, category=category)  ########## Creates document object


        ##### Creates and adds headline Sentence Objecct To document
        if headline:
            headline_sentence = Sentence(current_doc, headline)
            populate_token_list(headline_sentence, headline)
            current_doc.headline = headline_sentence
        ##########################################################


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

        current_topic= Topic(topic_id=topic_id,docsetA_id=docsetA_id) ########### Creates topic object


        ########## Creates and adds narrative Sentence Object To Topic
        title_sentence=Sentence(current_topic,title)
        populate_token_list(title_sentence, title)
        current_topic.title=title_sentence
        ##########################################################


        ############# Creates and adds narrative Sentence Object To Topic
        if narrative:
            narrative_sentence = Sentence(current_topic, narrative)
            populate_token_list(narrative_sentence, narrative)
            current_topic.narrative=narrative_sentence
        ##########################################################


        populate_document_list(current_topic, docsetA, doc_ret)

        current_topic.compute_tf_idf()

        topics.append(current_topic)

    return topics

###############################