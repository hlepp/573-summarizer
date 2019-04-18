#!/opt/python-3.6/bin/python3
# TODO: make sure this hash-bang runs correctly (double-check what we each use)

"""TODO: put a description of this file"""

# TODO: import statements go here (sys, etc.)
from data_input import get_data
from content_selection import select_content
from info_ordering import order_info
from content_realization import realize_content
from evaluation import eval_summary
from sys import argv
# TODO: probably import Topic, Document, Sentence classes


if __name__ == '__main__':
	
	# TODO: modify this to accept command line arguments
	# which will have a configuration file for which data to read in
	# that is then passed into get_data

	# Read in data
	# this returns a 1D array of Topic objects
	# (each of which contains a list of Document objects)
	topics = get_data(argv[1]) # TODO: modify to accept a config file of data source


	# Content Selection
	# identifies salient sentences & ranks them
	# & chooses up to 100 words (using full sentences)
	# returns a 2D array of sentences for the summary for each topic
	# and a 2D array of associated dates for each setence for each topic
	summaries_sent, sent_dates = select_content(topics)


	# Information Ordering
	# orders sentences by date
	# returns a 2D array of selected sentences in order for each topic
	summaries_in_order = order_info(summaries_sent, sent_dates)

	# Content Realization
	# process sentences to make well-formed
	# return a 2D array of well-formed sentences in order for each topic
	summaries = realize_content(summaries_in_order)

	# Write summary to file for each topic
	summary_files = write_summary_files(summary)


	# Evaluate summaries for each topic
	# Run ROUGE-1 & ROUGE-2 on the summary
	# TODO: modify this file to get ROUGE passed in
	# (or should it be hard-coded?)
	eval_summary(summary_files)




