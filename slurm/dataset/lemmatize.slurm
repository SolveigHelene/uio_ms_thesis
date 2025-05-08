#!/bin/bash



module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all
module --force swap StdEnv Zen2Env
module load nlpl-stanza/1.8.1-foss-2022b-Python-3.10.8
module load nlpl-nlptools/02-foss-2022b-Python-3.10.8

if [ -z "$1" ]; then
  echo "Please provide the language as an argument."
  exit 1
fi

if [ -z "$2" ]; then
  echo "Please provide stanza language id."
  exit 1
fi

LANGUAGE=$1
STANZA_LANG=$2

python3 lemmatize.py \
--input_file /cluster/work/users/solvehw/datasets/${LANGUAGE}.txt.gz \
--output_file /cluster/work/users/solvehw/datasets/lemmatized_${LANGUAGE}.txt \
--lang_id ${STANZA_LANG} \