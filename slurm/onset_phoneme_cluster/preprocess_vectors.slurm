#!/bin/bash

module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all
module load nlpl-nlptools/03-foss-2022b-Python-3.10.8

languages=("Dutch" "English" "German" "Italian" "Spanish" "Swedish")

for lang in "${languages[@]}"; do
    file_path_words="/cluster/work/users/solvehw/ipa_word_lists/ipa_${lang}.txt"
    phonemes="/cluster/work/users/solvehw/correlations/first_phonemes/${lang}/phoneme_keys.txt"
    file_path_sem="/cluster/work/users/solvehw/filtered_sem/${lang}.npy"
    file_path_phon="/cluster/work/users/solvehw/siamese_emb/siamese_${lang}.npy"
    base="/cluster/work/users/solvehw/correlations/first_phonemes/${lang}"

    echo ""
    echo "Processing language: $lang"
    
    python3 preprocess_vectors.py \
    --file_path_phon_words "$file_path_words" \
    --file_path_first_phonemes "$phonemes" \
    --file_path_sem_vectors "$file_path_sem"\
    --file_path_phon_vectors "$file_path_phon"\
    --lang "$lang" \
    --file_path_base "$base"
done
