import os
import logging
import argparse

import numpy as np


logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def find(file_path_ipa, file_path_ort, file_path_sem_vec, file_path_phon_vec, output_file_base):


    len_dict = {}

    with open(file_path_ipa, 'r') as file, open(file_path_ort, "r") as f:
        ipa_list = [word.strip().replace(" ", "") for word in file.readlines()]
        ortho_list = [word.strip() for word in f.readlines()]
        assert len(ipa_list) == len(ortho_list)
    
    sem_vec = np.load(file_path_sem_vec)
    phon_vec = np.load(file_path_phon_vec)


        
    logger.info("Going through list and saving based on lenght")
    for ipa, word, sem, phon in zip(ipa_list, ortho_list, sem_vec, phon_vec):
        length = len(ipa) 
        length_ort = len(word)

       # don't keep words than are shorter than two phonemes and one orthographic character
        if 2 <= length and length_ort > 1:
            if length not in len_dict:
                len_dict[length] = []
            len_dict[length].append((ipa, word, sem, phon))

    for length, word_data in len_dict.items():
        output_path = os.path.join(output_file_base, f"word_length_{length}")
        os.makedirs(output_path, exist_ok=True)
        
        ipa_out = []
        ortho_out = []
        sem_out = []
        phon_out = []

        for ipa, word, sem, phon in word_data:
            ipa_out.append(ipa)
            ortho_out.append(word)
            sem_out.append(sem)
            phon_out.append(phon)

        logger.info(f"Saving different file for length {length}")
        # Save as files
        with open(os.path.join(output_path, "ipa_words.txt"), "w") as f:
            f.write("\n".join(ipa_out))
        with open(os.path.join(output_path, "ort_words.txt"), "w") as f:
            f.write("\n".join(ortho_out))
        np.save(os.path.join(output_path, "sem.npy"), np.array(sem_out))
        np.save(os.path.join(output_path, "phon.npy"), np.array(phon_out))

    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path_ipa", type=str, required=True, help="Path to the file with IPA words")
    parser.add_argument("--file_path_ort", type=str, required=True, help="Path to the file with orthographic words")
    parser.add_argument("--file_path_sem_vec", type=str, required=True, help="Path to the semantic vectors file")
    parser.add_argument("--file_path_phon_vec", type=str, required=True, help="Path to the phonetic vectors file")
    parser.add_argument("--output_file_base", type=str, required=True, help="Base path for the output files")
    args = parser.parse_args()

    find(args.file_path_ipa, args.file_path_ort, args.file_path_sem_vec, args.file_path_phon_vec, args.output_file_base)