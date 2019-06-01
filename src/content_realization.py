#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Content realization for multi-document text summarization that generates well-formed sentences."""

__author__ = "Amina Venton, Shannon Ladymon"
__email__ = "aventon@uw.edu, sladymon@uw.edu"

import spacy
from spacy.attrs import IS_PUNCT, POS, TAG, DEP, ENT_TYPE, HEAD, ID, ORTH
from spacy.tokens import Doc
import numpy
import sys
import re
import time

def get_spaces(new_doc_np_array):
    """
    This function takes in a new numpy array of the Doc and
    re-calculates which token in the Doc should be followed by a space.
    The spaces in the new Doc only affect punctuation.
    Ex: "Hello ! "-> "Hello!"
 
    A list of boolean values are returned for each token
    indicating whether each word has a subsequent space
    tokens -> ["Hello", "!"]
    spaces bool -> [False, False]

    """

    # List to get indices of space before punctuation
    indices_of_space_before_punc = []

    # Iterate through each attribute, find the punctuation token index (minus one), add to list
    for array_index, array in enumerate(new_doc_np_array):

        # The first token attribute in each array IS_PUNCT, attribute value = 1 if true
        if array[0] == 1:
            # Space will be one position before punctuation index
            indices_of_space_before_punc.append(array_index - 1)

    # A list of boolean values indicating whether each word has a subsequent space
    # False if the token is followed by punctuation or the last token, true otherwise
    spaces = [False if any(x == i for x in indices_of_space_before_punc) or i == new_doc_np_array.shape[0] - 1
              else True for i in range(new_doc_np_array.shape[0])]

    # Return boolean values
    return spaces


def remove_subtree(doc, indices_to_remove_subtree):
    """
    This function removes a subtree given a spaCy Doc object and
    the indices of each token to remove.

    The Doc is converted to a numpy array shape (N, M), where N is
    the length of the Doc (in tokens) and M is a sequence of attributes,
    in order to remove a given subtree.

    A new Doc object is created with given attributes and words and
    spaces are also corrected. The new Doc is returned.

    """

    # Create array with token attributes for original doc
    original_doc_np_array = doc.to_array([IS_PUNCT, POS, TAG, DEP, ENT_TYPE, HEAD, ID, ORTH])

    # Create new array with the removed subtree from the original array
    new_doc_np_array = numpy.delete(original_doc_np_array, indices_to_remove_subtree, axis=0)

    # Tokens with punctuation will now be separated -> "that ,"
    # Get correct spaces for the new Doc object
    spaces = get_spaces(new_doc_np_array)

    # Create new spaCy Doc object
    new_doc = Doc(doc.vocab, words=[token.text for token_index, token in enumerate(doc) if
                                    token_index not in indices_to_remove_subtree], spaces=spaces)

    # Load token attributes to new spaCy Doc object
    new_doc.from_array([IS_PUNCT, POS, TAG, DEP, ENT_TYPE, HEAD, ID, ORTH], new_doc_np_array)

    return new_doc


def trim_sentence(doc, dependency_type):

    """
    This function trims a sentence given a spaCy Doc and
    the string dependency type corresponding to the clause to be removed.
    :param dependency_type: "appos", "acl", "relcl", "advcl"

    The new sentence with clause removed is returned in a spaCy Doc.
    If no dependency found original Doc is returned.

    """

    # Flag that indicates if dependency found, trim the sentence
    dependency_found = False

    # List of clause tokens in the subtree
    tokens_in_subtree = []

    # List of indices of clause tokens in the Doc
    indices_to_remove_subtree = []

    # Iterate through the doc to get the dep clause subtree
    for token in doc:

        # Check for dependency label
        if token.dep_ == dependency_type:
            dependency_found = True

            # If found, get the subtree- all tokens of the clause
            for subtree_token in token.subtree:

                # Get the preceding punct token in the subtree -> ( ABC
                # Exclude first token
                if doc[0] != subtree_token:
                    if subtree_token.nbor(-1).text in (",", "(",):
                        tokens_in_subtree.append(subtree_token.nbor(-1))

                # Get the token in the subtree -> ABC
                tokens_in_subtree.append(subtree_token)

                # Get the following punct token in the subtree -> ABC )
                # Exclude last token
                if doc[-1] != subtree_token:
                    if subtree_token.nbor(1).text in (",", ")"):
                        tokens_in_subtree.append(subtree_token.nbor(1))


    # Trim the sentence
    if dependency_found:
        # Get the indices of the tokens to be removed in the Doc object
        for index, token in enumerate(doc):
            if token in tokens_in_subtree:
                indices_to_remove_subtree.append(index)
        
        # Remove the subtree and create new Doc
        new_doc = remove_subtree(doc, indices_to_remove_subtree)
        
        print("***Dependency Found***")
        print("Clean sent: {}".format(doc[:].text))
        print("New Sentence: {}".format(new_doc[:].text))

        # Return the new trimmed Doc in a string
        return new_doc[:].text

    # Return original doc in a string if no dependency found
    return doc[:].text


def clean_sentence(original_sent, remove_header, remove_parens, remove_quotes):
    """
    Cleans a sentence by fixing newlines and spaces, and optionally by removing
    the news header, parenthetical information, and unpaired quotes.
   
    Args:
        original_sent: original sentence string
        remove_header: True if the header should be removed from the sentence
        remove_parens: True if parenthetical information should be removed from the sentence
        remove_quotes: True if unpaired quotes should be removed from the sentence

    Returns:
        The cleaned sentence.

    """

    # Start with the clean_sent set to the original as a default
    clean_sent = original_sent

    # Clean sentence by removing the header, if specified
    if remove_header:

        # Matches headers of form "NEW YORK, July 1 (AP) --" and "NEW YORK _ " and "NEW YORK (AP) _"
        header_1_re = re.compile(r"^\s+([A-Z]+\s*)+,?(\s*([a-zA-Z]+\.?)*\s*[0-9]*\s*)?(\([a-zA-Z]+\))?\s*((--)|(_))\s*")

        # Matches headers of form "BC-FLA (Tampa) --" and "COLO-SCHOOL-SHOOTING (Littleton, Colo.) _"
        header_2_re = re.compile(r"^([A-Z]+-?)+\s\([a-zA-Z,\s\.]+\)\s((--)|(_))")

        # Matches headers of form "(AP) --" and "_"
        header_3_re = re.compile(r"^(\([a-zA-Z]+\))?\s*((--)|(_))\s*")

        clean_sent = header_1_re.sub(r"", clean_sent)
        clean_sent = header_2_re.sub(r"", clean_sent)
        clean_sent = header_3_re.sub(r"", clean_sent)

    # Clean sentence by removing any parenthetical information, if specified
    if remove_parens:
        # Matches any parenthetical information
        parens_re = re.compile(r"\([^\)]+\)")
        clean_sent = parens_re.sub(r"", clean_sent)

    # Clean sentence by removing any unpaired quotes, if specified
    if remove_quotes:
        # Matches for different types of quotes (", ``, '') to find unpaired quotes
        quote_re = re.compile(r"\"")
        backtick_re = re.compile(r"``")
        apostrophe_re = re.compile(r"''")

        # Get the number of quotes to see if unmatched
        num_quotes = len(quote_re.findall(clean_sent))
        num_backticks = len(backtick_re.findall(clean_sent))
        num_apostrophes = len(apostrophe_re.findall(clean_sent))

        if num_quotes == 1:
            # Remove unpaired " mark
            clean_sent = quote_re.sub(r"", clean_sent)
        if num_backticks == 1 and num_apostrophes == 0:
            # Remove `` (open quote) unpaired with '' (closing quote)
            clean_sent = backtick_re.sub(r"", clean_sent)
        if num_apostrophes == 1 and num_backticks == 0:
            # Remove '' (closing quote) unpaired with `` (open quote)
            clean_sent = apostrophe_re.sub(r"", clean_sent)

    # Remove any extra newlines
    clean_sent = clean_sent.replace("\n", " ").strip()

    # Remove any extra spaces
    spaces_re = re.compile(r" {2,}")
    clean_sent = spaces_re.sub(r" ", clean_sent)

    return clean_sent


def get_compressed_sentences(original_sent, spacy_parser, remove_header, remove_parens, remove_quotes, remove_appos, remove_advcl, remove_relcl, remove_acl):
    """
    This function performs sentence compression given an sentence string and rule-type.
   
    Args:
        original_sent: original sentence string
        spacy_parser: spaCy parser model
        remove_header: True if the header should be removed from the sentence
        remove_parens: True if parenthetical information should be removed from the sentence
        remove_quotes: True if unpaired quotes should be removed from the sentence
        remove_appos: True if appositional modifier should be removed from the sentence
        remove_advcl: True if adverbial clause modifier should be removed from the sentence
        remove_relcl: True if relative clause modifier should be removed from the sentence
        remove_acl: True if a finite or non-finite clausal modifier shoule be removed from the sentence

    Returns:
        sentences_list: list of compressed versions of the original sentence
    """
    start_time = time.time()
    # List of sentence strings to return
    sentences_list = []  

    # Get clean version of the sentence
    clean_sent = clean_sentence(original_sent, remove_header, remove_parens, remove_quotes)

    # Add this cleaned version of the sentence to the list
    sentences_list.append(clean_sent)


    # Remove branches of syntax tree from spaCy Doc
    if remove_appos or remove_advcl or remove_relcl or remove_acl:
        doc = spacy_parser(clean_sent)
    
        # Remove appositional modifiers (appos)
        if remove_appos:
            new_doc_sent = trim_sentence(doc, "appos")
            sentences_list.append(new_doc_sent)

        # Remove adverbial clausal modifiers (advcl)
        if remove_advcl:
            new_doc_sent = trim_sentence(doc, "advcl")
            sentences_list.append(new_doc_sent)

        # Remove relative clauses (relcl)
        if remove_relcl:
            new_doc_sent = trim_sentence(doc, "relcl")
            sentences_list.append(new_doc_sent)

        # Remove clausal modifiers (acl)
        if remove_acl:
            new_doc_sent = trim_sentence(doc, "acl")
            sentences_list.append(new_doc_sent) 

    
        #print("New doc sent: {}".format(new_doc_sent))
    
    end_time = time.time()
    #print("Total runtime for compressed_sent: {}".format(end_time - start_time))

#    if "BURBANK, Calif. (AP)" in original_sent:
#        sys.exit("TESTING: done up to Burbank sentence")

    return sentences_list


"""

#TODO: testing remove when done
if __name__ == '__main__':

    spacy_parser = spacy.load('/home/longwill/en_core_web_md/en_core_web_md-2.1.0')
    
    original_sent = "I wanted to tell you that, Bill, John's cousin, lives in Seattle."
    #original_sent = "The accident happened as the night was falling."
    #original_sent = "I saw the book which you bought."
    #original_sent = "I admire that you are honest."
    get_compressed_sentences(original_sent, spacy_parser, remove_header = False, remove_parens = False, remove_quotes = False, remove_appos = True, remove_advcl = True, remove_relcl = True, remove_acl = True)


"""
