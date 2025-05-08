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
  echo "Please provide the type of embedding."
  exit 1
fi

LANGUAGE=$1
TYPE=$2

echo ""
echo "Processing language: ${LANGUAGE}, computing pairwise: ${TYPE}"

python3 compute_pairwise.py \
--input_file /cluster/work/users/solvehw/removed_duplicates/${LANGUAGE}_${TYPE}.npy \
--output_file /cluster/work/users/solvehw/removed_duplicates/pairwise_distances/${LANGUAGE}_${TYPE}_all.npy

