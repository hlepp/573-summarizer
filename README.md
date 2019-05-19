# 573-summarizer
LING 573 Group Project
Nutshell topic-focused multi-document extractive text summarizer

## Getting Started
Run D3.cmd on condor in order to run the Nutshell text summarizer.
This will create the summary files in the outputs/D3/ directory, and the ROUGE results in /results/ directory.

This will run the current best version of Nutshell, with the following parameters:

```
input_file = 
output_folder = D3
stemming = True
lower = False
idf_type = 'smooth_idf'
tf_type = 'term_frequency'
d = 0.2
intersent_threshold = 0.1
summary_threshold = 0.3
epsilon = 0.04
mle_lambda = 0.6
k = 9
min_sent_len = 3
include_narrative = False
bias_formula = 'rel'
intersent_formula = 'cos'
info_order_type = 'entity'
num_permutations = 5
```

## Authors
Shannon Ladymon, sladymon@uw.edu

Amina Venton, aventon@uw.edu

Haley Lepp, hlepp@uw.edu 

Ben Longwill, longwill@uw.edu

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) for details

