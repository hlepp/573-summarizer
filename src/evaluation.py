#!/opt/python-3.6/bin/python3

def eval_summary(summaries):
	# TODO: Loop through summary files and compare to equivalent XML data
        # TODO: Run ROUGE command within loop
        # TODO: print to D2_rouge_scores.out
	print()

if __name__ == '__main__':
	ROUGE_DATA_DIR = '/dropbox/18-19/573/code/ROUGE/data'
	CONFIG_FILE_WITH_PATH = '/ROUGE/revised_config'
	COMMAND = './ROUGE/ROUGE-1.5.5.pl -e ' + ROUGE_DATA_DIR + ' -a -n 2 -x -m -c 95 -r 1000 -f A -p 0.5 -t 0 -l 100 -s -d ' + CONFIG_FILE_WITH_PATH
