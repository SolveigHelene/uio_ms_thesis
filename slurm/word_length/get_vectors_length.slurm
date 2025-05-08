#!/bin/bash

module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all
module load nlpl-nlptools/02-foss-2022b-Python-3.10.8

if [ -z "$1" ]; then
  echo "Please provide the language as an argument."
  exit 1
fi

LANGUAGE=$1

echo ${LANGUAGE}

python3 get_vectors_length.py \
--words /cluster/work/users/solvehw/removed_duplicates/word_length/${LANGUAGE}/${LANGUAGE}_indefinite_nominative_singular_forms.txt \
--word_list /cluster/work/users/solvehw/removed_duplicates/${LANGUAGE}_ort.txt \
--ipa_list removed_duplicates/${LANGUAGE}_ipa.txt \
--phon /cluster/work/users/solvehw/removed_duplicates/${LANGUAGE}_phon.npy \
--sem /cluster/work/users/solvehw/removed_duplicates/${LANGUAGE}_sem.npy \
--output_base /cluster/work/users/solvehw/removed_duplicates/word_length/${LANGUAGE}