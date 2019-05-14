#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Runs ROUGE evaluation on the summary file results."""


__author__ = 'Haley Lepp'
__email__ = 'hlepp@uw.edu'


import os
import subprocess
from subprocess import Popen, PIPE

def change_xml(output_folder):
    default_xml = open('src/ROUGE/revised_config.xml', 'r')
    new_xml = open('src/ROUGE/revised_config_new.xml', 'w+')
    if not os.path.exists('outputs/' + output_folder):
        os.mkdir(output_folder)
    for line in default_xml:
        if line == 'outputs/D2\n':
            new_xml.write('outputs/' + output_folder + '\n')
        else:
            new_xml.write(line)


def eval_summary(output_folder):
    change_xml(output_folder)
    ROUGE_DATA_DIR = '/dropbox/18-19/573/code/ROUGE/data'
    CONFIG_FILE_WITH_PATH = 'src/ROUGE/revised_config_new.xml'
    # CONFIG_FILE_WITH_PATH = 'src/ROUGE/revised_config.xml'
    COMMAND = 'src/ROUGE/ROUGE-1.5.5.pl -e ' + ROUGE_DATA_DIR \
        + ' -a -n 2 -x -m -c 95 -r 1000 -f A -p 0.5 -t 0 -l 100 -s -d ' \
        + CONFIG_FILE_WITH_PATH
    stdout = subprocess.check_output(COMMAND.split())
    output_file = "results/" + output_folder + "_rouge_scores.out"
    results = open(output_file, 'w+')
    results.write(stdout.decode())
    results.close()
