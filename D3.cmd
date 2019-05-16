universe = vanilla
executable = /usr/bin/python3
getenv = true
#input =
output = cmd_out_3
error = error_3
log = log_3
arguments = src/text_summarizer.py /dropbox/18-19/573/Data/Documents/devtest/GuidedSumm10_test_topics.xml D3_test False False smooth_idf term_frequency 0.7 0.0 0.5 0.1 0.6 20 5 False cos cos
transfer_executable = false
queue
