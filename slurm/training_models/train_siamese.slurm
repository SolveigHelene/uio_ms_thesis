#!/bin/bash


module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all
module --force swap StdEnv Zen2Env
module load nlpl-nlptools/03-foss-2022b-Python-3.10.8
module load nlpl-python-candy/01-foss-2022b-Python-3.10.8 

module load nlpl-scikit-bundle/1.3.2-foss-2022b-Python-3.10.8
module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8

if [ -z "$1" ]; then
  echo "Please provide the language as an argument."
  exit 1
fi

LANGUAGE=$1

python3 /cluster/work/users/solvehw/siamese_network/train.py \
--data-file /cluster/work/users/solvehw/ipa_word_list/ipa_${LANGUAGE}.txt \
--train-batch-size 1 \
--val-batch-size 1 \
--n-epochs 2 \
--embs-outfile /cluster/work/users/solvehw/siamese_emb/siamese_${LANGUAGE}.txt \
--lang ${LANGUAGE}