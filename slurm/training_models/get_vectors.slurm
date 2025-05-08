#!/bin/bash

module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all
module load nlpl-nlptools/02-foss-2022b-Python-3.10.8


if [ -z "$1" ]; then
  echo "Please provide the language as an argument."
  exit 1
fi

LANGUAGE=$1

python3 get_vectors.py \
--input_file /cluster/work/users/solvehw/siamese_emb/siamese_${LANGUAGE}.txt \
--output_file /cluster/work/users/solvehw/siamese_emb/siamese_${LANGUAGE}.npy