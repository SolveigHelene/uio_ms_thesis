#!/bin/bash



module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all
module load nlpl-nlptools/03-foss-2022b-Python-3.10.8
module load nlpl-python-candy/01-foss-2022b-Python-3.10.8 
module load nlpl-scikit-bundle/1.3.2-foss-2022b-Python-3.10.8

if [ -z "$1" ]; then
  echo "Please provide the language as an argument."
  exit 1
fi

LANGUAGE=$1

# Run Python script with dynamic paths based on the language
python3 compute_pairwise_2.py \
--word_freq_dir /cluster/work/users/solvehw/frequency_list/frequency_${LANGUAGE}.txt \
--sem_vec_dir /cluster/work/users/solvehw/filtered_sem/${LANGUAGE}_sem.npy \
--phon_vec_dir /cluster/work/users/solvehw/siamese_emb/${LANGUAGE}_phon.npy \
--output_dir /cluster/work/users/solvehw/pairwise_distances \
--lang $LANGUAGE

