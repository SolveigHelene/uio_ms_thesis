#!/bin/bash

module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all
module load nlpl-nlptools/03-foss-2022b-Python-3.10.8


if [ -z "$1" ]; then
  echo "Please provide the language as an argument."
  exit 1
fi

LANGUAGE=$1

    arg("--input_file", type=str, required=True)
    arg("--output_file", type=str, required=True)
    arg("--epitran_code", type=str, required=True)

echo ""
echo "Processing language: $LANGUAGE"

python3 preprocess_sem_vec.py \
--input_file /cluster/work/users/solvehw/Ukrainian_word_list.txt \
--output_file /cluster/work/users/solvehw/Ukrainian_ipa_list.txt \
--epitran_code ukr-Cyrl \
