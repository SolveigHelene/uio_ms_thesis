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

if [ -z "$2" ]; then
  echo "Please provide the frequency."
  exit 1
fi

LANGUAGE=$1
FREQ=$2



echo ""
echo "Processing language: $lang"

python3 correlation_copy.py \
    --file_path_sem /cluster/work/users/solvehw/pairwise_distances/${LANGUAGE}_phon_${FREQ}.npy \
    --file_path_phon /cluster/work/users/solvehw/pairwise_distances/${LANGUAGE}_sem_${FREQ}.npy \
    --output_csv /cluster/work/users/solvehw/results_${FREQ}.csv \
    --lang $LANGUAGE
