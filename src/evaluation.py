#!/opt/python-3.6/bin/python3

import subprocess
from subprocess import Popen, PIPE

def eval_summary():
	ROUGE_DATA_DIR = '/dropbox/18-19/573/code/ROUGE/data'
	CONFIG_FILE_WITH_PATH = 'ROUGE/revised_config.xml'
	COMMAND = './ROUGE/ROUGE-1.5.5.pl -e ' + ROUGE_DATA_DIR + ' -a -n 2 -x -m -c 95 -r 1000 -f A -p 0.5 -t 0 -l 100 -s -d ' + CONFIG_FILE_WITH_PATH
	stdout = subprocess.check_output(COMMAND.split())
	results=open('../results/D2_rouge_scores.out', 'w+')
	results.write(stdout)
	results.close()

eval_summary()


        
