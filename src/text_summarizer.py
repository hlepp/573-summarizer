#!/opt/python-3.6/bin/python3
# TODO: make sure this hash-bang runs correctly (double-check what we each use)

"""TODO: put a description of this file"""

# TODO: import statements go here (sys, etc.)
import get_data
import select_content
import order_info
import realize_content
import eval_summary



if __name__ == '__main__':
	
	# Read in data
	# TODO: Determine format of data returned
	# FOR NOW: return a 3D array of sentences from each doc from each topic set
	# TODO: modify this file to get the data passed in
	doc_data = get_data()


	# Content Selection
	# identifies salient sentences & ranks them
	# & chooses up to 100 words
	# TODO: figure out if we should have exactly 100 words,
	# or just full sentences up to that limit
	# returns a 2D array of sentences for the summary for each topic
	# and a 2D array of associated dates for each setence for each topic
	summaries_sent, sent_dates = select_content(doc_data)


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
	eval_summary(summary_files)




