import os
import logging
import argparse

import numpy as np

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Script used for getting semantic and phonetic embeddings for the different lengths

def main(words, word_list, ipa_list, phon_vectors, sem_vectors, output_base): 
    
    logger.info("Loading all words and vectors")
    with open(words, 'r', encoding="utf-8") as file:
        words_for_length = [line.strip() for line in file.readlines()] 
    
    with open(word_list, 'r') as f:
        all_words = [word.strip() for word in f.readlines()]
    
    with open(ipa_list, 'r') as f_ipa:
        all_words_ipa = [word.strip() for word in f_ipa.readlines()]
    
    phon = np.load(phon_vectors)
    sem = np.load(sem_vectors)
    logger.info("Starting sorting out words and vectors")
    
    keep_word = []
    keep_ipa = []
    keep_phon = []
    keep_sem = []

    for word, ipa, p_vec, s_vec in zip(all_words, all_words_ipa, phon, sem):
        if word in words_for_length:
            keep_word.append(word)
            keep_ipa.append(ipa)
            keep_phon.append(p_vec)
            keep_sem.append(s_vec)

    logger.info(f"Words to keep: {len(keep_word)}")
    # logger.info(f"Phon to keep: {len(phon_to_keep)}, sem to keep: {len(sem_to_keep)}")
    # assert len(phon_to_keep) == len(sem_to_keep) == len(words_for_length)
    # assert len(words_for_length) == len(sem_to_keep)
    

    logger.info("Saving vectors")
    output_phon = os.path.join(output_base, f"sorted_phon.npy")
    output_sem = os.path.join(output_base, f"sorted_sem.npy")
    output_words = os.path.join(output_base, f"sorted_words.txt")
    output_ipa = os.path.join(output_base, f"sorted_ipa.txt")


    np.save(output_phon, keep_phon)
    np.save(output_sem, keep_sem)

    with open(output_words, 'w', encoding='utf-8') as f_ort_out:
        f_ort_out.write('\n'.join(keep_word))

    with open(output_ipa, 'w', encoding='utf-8') as f_ipa_out:
        f_ipa_out.write('\n'.join(keep_ipa))
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--words", type=str, required=True, help="directory to file with words for current length")
    parser.add_argument("--word_list", type=str, required=True, help="directory to file with all words written orthographicly")
    parser.add_argument("--ipa_list", type=str, required=True, help="directory to file with all words written phoneticly")
    parser.add_argument("--phon_vectors", type=str, required=True, help="directory to phonetic embeddings")
    parser.add_argument("--sem_vectors", type=str, required=True, help="directory to semantic embeddings")
    parser.add_argument("--output_base", type=str, required=True, help="base directory to where the new filtered embeddings should be saved")
    args = parser.parse_args()
    
    main(args.words, args.word_list, args.ipa_list, args.phon_vectors, args.sem_vectors, args.output_base)