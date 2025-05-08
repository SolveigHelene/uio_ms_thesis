#!/bin/bash

module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all
module --force swap StdEnv Zen2Env
module load nlpl-stanza/1.8.1-foss-2022b-Python-3.10.8


module swap nlpl-huggingface-hub/0.19.4-foss-2022b-Python-3.10.8 nlpl-huggingface-hub/0.23.5-foss-2022b-Python-3.10.8
module swap nlpl-datasets/2.17.1-foss-2022b-Python-3.10.8 nlpl-datasets/2.21.0-foss-2022b-Python-3.10.8

module load nlpl-nlptools/03-foss-2022b-Python-3.10.8

python3 combine_dataset.py --input_dir /cluster/shared/nlpl/data/corpora/conll17/udpipe/English --output_dir /cluster/work/users/solvehw/datasets/English_word_pos.txt