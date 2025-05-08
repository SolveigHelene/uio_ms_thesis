#!/bin/bash


module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all
module use Python/3.12.3-GCCcore-13.3.0
module load Python/3.10.8-GCCcore-12.2.0
module load nlpl-gensim/4.3.2-foss-2022b-Python-3.10.8

if [ -z "$1" ]; then
  echo "Please provide the language as an argument."
  exit 1
fi

LANGUAGE=$1

python3 train_skipgram.py --corpus /cluster/work/users/solvehw/datasets/${LANGUAGE}.txt.gz --lang ${LANGUAGE}


