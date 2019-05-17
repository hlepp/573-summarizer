#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Runs ROUGE evaluation on the summary file results."""

__author__ = 'Haley Lepp'
__email__ = 'hlepp@uw.edu'

import os
import subprocess
import config_changer
from os.path import abspath
from subprocess import Popen, PIPE


def edit_xml_dev(output_folder):
    default_xml = open('src/ROUGE/revised_config.xml', 'r')
    config_file = "src/ROUGE/revised_config" + output_folder + ".xml"
    new_xml = open(config_file, "w+")    
    if not os.path.exists('outputs/' + output_folder):
        os.mkdir(output_folder)
    for line in default_xml:
        if line == 'outputs/D2\n':
            new_xml.write('outputs/' + output_folder + '\n')
        else:
            new_xml.write(line)
    return config_file


def edit_xml_train(output_folder):
    config_file = "src/ROUGE/revised_config" + output_folder + ".xml"
    new_xml = open(config_file, "w+")
    peers = os.listdir('/dropbox/18-19/573/Data/models/training/2009')
    peers = sorted(peers)
    print(peers)
    if not os.path.exists('outputs/' + output_folder):
        os.mkdir(output_folder)
    with open('src/ROUGE/revised_config.xml', 'r') as f:
        default_xml = f.readlines()
    peers_counter = 0
    for line in default_xml:
        if line  == 'outputs/D2\n':
            new_xml.write('outputs/' + output_folder + '\n')
        elif "D10" in line:
            line = line.replace("D10", "D09")
            if "<M ID=\"" in line:
                peer = peers[peers_counter] 
                letter_1 = peer[len(peer) - 1]
                # letter_2 = peer[len(peer) - 3]
                # letter_3 = peer[6]
                # label = peer[:5]
                line = '<M ID=\"' + letter_1 + '\">' + peer + '</M>\n'
                # line = '<M ID=\"' + letter_1 + '\">' + str(label) + '-' + letter_3 + '.M.100.' + letter_2 + '.' + letter_1 + '</M>\n'
            new_xml.write(line)
            peers_counter +=1
        elif line == '/dropbox/18-19/573/Data/models/devtest/\n':
            new_xml.write('/dropbox/18-19/573/Data/models/training/2009\n')
        else:
            new_xml.write(line)
    return config_file


def eval_summary(output_folder, data_type):
    if data_type == 'train':
        CONFIG_FILE_WITH_PATH = edit_xml_train(output_folder)
    elif data_type == 'dev':
        CONFIG_FILE_WITH_PATH = edit_xml_dev(output_folder)
    else:
        raise ValueError('First argument must be train or dev')  
    ROUGE_DATA_DIR = '/dropbox/18-19/573/code/ROUGE/data'
    train_model = '/dropbox/18-19/573/Data/models/training/2009'
    COMMAND = 'src/ROUGE/ROUGE-1.5.5.pl -e ' + ROUGE_DATA_DIR \
        + ' -a -n 2 -x -m -c 95 -r 1000 -f A -p 0.5 -t 0 -l 100 -s -d ' \
        + CONFIG_FILE_WITH_PATH
    stdout = subprocess.check_output(COMMAND.split())
    output_file = 'results/' + output_folder + '_rouge_scores.out'
    results = open(output_file, 'w+')
    results.write(stdout.decode())
    results.close()
