#!/bin/bash


module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all
module load nlpl-nlptools/03-foss-2022b-Python-3.10.8
module load nlpl-python-candy/01-foss-2022b-Python-3.10.8 
module load nlpl-scikit-bundle/1.3.2-foss-2022b-Python-3.10.8

languages=("Dutch" "English" "German" "Italian" "Spanish" "Swedish")

for lang in "${languages[@]}"; do
    base_path="/cluster/work/users/solvehw/correlations/first_phonemes/${lang}"
    output="/cluster/work/users/solvehw/correlations/first_phonemes/${lang}/correlations_siamese.csv"
    phoneme_list="/cluster/work/users/solvehw/correlations/first_phonemes/${lang}/phoneme_keys.txt"

    echo ""
    echo "Processing language: $lang"
    
    python3 correlation_phonemes.py \
        --phoneme_list "$phoneme_list" \
        --base_path "$base_path" \
        --output_csv "$output" 
done
