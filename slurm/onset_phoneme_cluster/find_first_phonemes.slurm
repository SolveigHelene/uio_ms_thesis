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


python3 find_first_phonemes.py \
--ipa_file /cluster/work/users/solvehw/removed_duplicates/${LANGUAGE}_ipa.txt \
--ortho_file /cluster/work/users/solvehw/removed_duplicates/${LANGUAGE}_ort.txt \
--sem_vector /cluster/work/users/solvehw/removed_duplicates/${LANGUAGE}_sem.npy \
--phon_vector /cluster/work/users/solvehw/removed_duplicates/${LANGUAGE}_phon.npy \
--output_file /cluster/work/users/solvehw/removed_duplicates/phonestheme/${LANGUAGE} \
--lang ${LANGUAGE}

