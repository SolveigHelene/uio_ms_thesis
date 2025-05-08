#!/bin/bash

module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all
module load nlpl-nlptools/03-foss-2022b-Python-3.10.8


if [ -z "$1" ]; then
  echo "Please provide the language as an argument."
  exit 1
fi

LANGUAGE=$1


echo ""
echo "Processing language: $LANGUAGE"

python3 preprocess_sem_vec.py \
--file_path_sem /cluster/work/users/solvehw/frequency_list/frequency_${LANGUAGE}.txt \
--file_path_phon /cluster/work/users/solvehw/ipa_word_lists/ort_${LANGUAGE}.txt \
--file_path_vectors /cluster/work/users/solvehw/skipgram_models/skipgram_${LANGUAGE}.model.vectors.npy \
--lang ${LANGUAGE}




