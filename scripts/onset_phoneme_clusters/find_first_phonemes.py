import os
import logging
import argparse

import numpy as np

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def find(ipa_words, ortho_words, phon_vector, sem_vector, output_file, lang):
    logger.info("Loading words")

    with open(ipa_words, "r") as f_ipa, open(ortho_words, "r") as f_ortho:
        ipa_list = [word.strip().replace(" ", "") for word in f_ipa.readlines()]
        ortho_list = [word.strip() for word in f_ortho.readlines()]

    if len(ipa_list) != len(ortho_list):
        logger.error("Mismatch between IPA and orthographic word lists")
        return
    
    logger.info("Loading vectors")
    sem_vec = np.load(sem_vector)
    phon_vec = np.load(phon_vector)

    mapping = list(zip(ipa_list, ortho_list, sem_vec, phon_vec))

    first_phonemes = {}

    logger.info("Going through mapping")
    for ipa, ort, sem, phon in mapping:
        if len(ipa) == 1 or len(ipa) >= 50:
            continue
        first = ipa[:2]

        if first not in first_phonemes:
            first_phonemes[first] = []
        if (ipa, ort, sem, phon) not in first_phonemes[first]:
            first_phonemes[first].append((ipa, ort, sem, phon))
    
    logger.info("Filtering phoneme groups with more than 150 words")
    filtered_phonemes = {k: v for k, v in first_phonemes.items() if len(v) > 150}
    if not os.path.exists(output_file):
        os.makedirs(output_file, exist_ok=True)

    logger.info("Going through and saving to files")
    for phones in filtered_phonemes.keys():
        folder = os.path.join(output_file, phones)
        os.makedirs(folder, exist_ok=True)     

        ortho_file = os.path.join(folder, "words_ort.txt")
        ipa_file = os.path.join(folder, "words_ipa.txt")
        sem_file = os.path.join(folder, "sem.npy")
        phon_file = os.path.join(folder, "phon.npy")  

        ipa_words_list = []
        ortho_words_list = []
        sem_vectors_list = []
        phon_vectors_list = []

        for ipa, ort, sem, phon in filtered_phonemes[phones]:
            ipa_words_list.append(ipa)
            ortho_words_list.append(ort)
            sem_vectors_list.append(sem)
            phon_vectors_list.append(phon)

        # Save orthographic words
        with open(ortho_file, "w") as f_ortho:
            f_ortho.write("\n".join(ortho_words_list) + "\n")
        
        with open(ipa_file, "w") as f_ipa:
            f_ipa.write("\n".join(ipa_words_list) + "\n")

        # Save semantic and phonetic vectors
        np.save(sem_file, np.array(sem_vectors_list))
        np.save(phon_file, np.array(phon_vectors_list))
    
    phoneme_keys_file = os.path.join(output_file, "phoneme_keys.txt")
    with open(phoneme_keys_file, "w") as file:
        file.writelines(f"{phones}\n" for phones in filtered_phonemes.keys())

    logger.info(f"Output written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipa_file", type=str, required=True, help="Path to the IPA words file")
    parser.add_argument("--ortho_file", type=str, required=True, help="Path to the orthographic words file")
    parser.add_argument("--sem_vector", type=str, required=True, help="Path to the semantic vectors")
    parser.add_argument("--phon_vector", type=str, required=True, help="Path to the phonetic vectors")
    parser.add_argument("--output_file", type=str, required=True, help="Directory to save the output")
    parser.add_argument("--lang", type=str, required=True, help="Language identifier")
    args = parser.parse_args()

    find(args.ipa_file, args.ortho_file, args.phon_vector, args.sem_vector, args.output_file, args.lang)


