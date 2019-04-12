#!/opt/python-3.6/bin/python3
# -*- coding: utf-8 -*-

"""Content Selector for multi-document text summarization that ranks and chooses the sentences to use in the summary."""

__author__ = "Shannon Ladymon"
__email__ = "sladymon@uw.edu"

# TODO: Possibly use nltk tokenizer to tokenize words, but this separates spaces...
# TODO: If used, add info on nltk tokenizer into project report
#from nltk.tokenize import word_tokenize



# TODO: delete after temporary testing
DOC_DATA = [
		[
			["This is sentence one in the first document.", "This is sentence two in the first document, which is a little longer.", "This is sentence three is the first document.", "This is sentence four in the first document."],
			["This is sentence one in the second document, which is even longer than the previous.", "This is sentence two in the second document, which is definitely the longest of all the sentences."],
			["This is sentence one in the third document, I lied, it's shorter.", "This is sentence two in the third document. It's okay, I guess.", "This is sentence three in the third document, which should be too long to add to the summary.", "4th_sent_3rd_doc okay-to-add-if-it-fits?"]
		],
		[
			["This is the only sentence in the only doc in the second topic."]
		]
	]

# TODO: Delete after temporary testing
DOC_DATES = [[20190425, 20190411, 20190428],[20190415]]

def select_content(doc_data):
	"""
	TODO: Update once we've figured out what input this function receives

	This is the function called from text_summarizer.py
	and should take in a 3D (?) array of [topics][docs][sentences]
	(NOTE: Needs to take in metadata like date as well!)
	and choose the best sentences for each topic, 
	returning a 2D array of topics and the chosen sentences
	as well as a 2D array of topics and the dates for those sentences
	(NOTE: This maybe should be modified to be some class/other data structure)
	"""

	# TODO: Delete after we get everything working with Ben's input
	# for now, it allows this program to run with some fake data
	doc_data = DOC_DATA
	doc_dates = DOC_DATES

	# A 2D array to hold the chosen sentences for each topic
	# [topic][sentences]
	summaries_sent = []
	
	# A 2D array to hold the associated dates for each sentence chosen
	# for each topic [topic][dates]
	sent_dates = []

	# Call a ranking method TODO: Change to a real algorithm later!
	# for now, just grab sentence from each doc while <= 100 words
	for topic_index in range(len(doc_data)):

		topic = doc_data[topic_index]

		# Arrays to chold the chosen sentences & dates
		# for this topic
		topic_chosen_sent = [] 
		topic_chosen_sent_dates = []
		
		# Keep track of number of words so far in summary
		summary_size = 0

		# Add sentences from the documents as long as there is room
		for doc_index in range(len(topic)):

			doc = topic[doc_index]
			date = doc_dates[topic_index][doc_index]

			for sent in doc:

				# TODO: consider other tokenization methods
				# Count the number of words (text with space in between)
				sent_len = sent.count(" ") + 1

				# Check if this sentence can be added to the summary
				# (if there is room) and do so if possible
				if summary_size + sent_len <= 100:
					summary_size += sent_len
					topic_chosen_sent.append(sent)
					topic_chosen_sent_dates.append(date)

		# Add this topic's summary sentences and dates
		# to return arrays
		summaries_sent.append(topic_chosen_sent)
		sent_dates.append(topic_chosen_sent_dates)

	# Return 2D arrays of chosen sentences and dates
	return summaries_sent, sent_dates


# TODO: delete after temporary testing
if __name__ == '__main__':

	summaries_sent, sent_dates = select_content(DOC_DATA)

	print("\nTESTING: summaries_sent={}".format(summaries_sent))
	print("TESTING: sent_dates={}".format(sent_dates))

