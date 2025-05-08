#!/bin/bash



module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all
module load nlpl-nlptools/03-foss-2022b-Python-3.10.8

if [ -z "$1" ]; then
  echo "Please provide the language as an argument."
  exit 1
fi


LANGUAGE=$1

python3 phon_count_emb.py \
--dataset /cluster/work/users/solvehw/ipa_word_lists/ipa_${LANGUAGE}.txt \
--lang ${LANGUAGE}