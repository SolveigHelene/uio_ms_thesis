#!/bin/bash


module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all
module load nlpl-nlptools/03-foss-2022b-Python-3.10.8

if [ -z "$1" ]; then
  echo "Please provide the language as an argument."
  exit 1
fi

LANGUAGE=$1

echo ${LANGUAGE}

python3 sort_based_length.py \
--file_path_ipa /cluster/work/users/solvehw/removed_duplicates/word_length/${LANGUAGE}/sorted_ipa.txt \
--file_path_ort /cluster/work/users/solvehw/removed_duplicates/word_length/${LANGUAGE}/sorted_words.txt \
--file_path_sem_vec /cluster/work/users/solvehw/removed_duplicates/word_length/${LANGUAGE}/sorted_sem.npy \
--file_path_phon_vec /cluster/work/users/solvehw/removed_duplicates/word_length/${LANGUAGE}/sorted_phon.npy \
--output_file_base /cluster/work/users/solvehw/removed_duplicates/word_length/${LANGUAGE}