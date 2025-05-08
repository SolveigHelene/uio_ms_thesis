import os
import logging
import argparse

import numpy as np
from scipy.spatial.distance import pdist

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_and_save_pairwise_distances(input_file, output_file, phoneme_list, type_of):

    with open(phoneme_list, 'r', encoding="utf-8") as file:
        first_phonemes = [line.strip() for line in file.readlines()]

    logger.info("Starting going through phonemes")
    for phoneme in first_phonemes:
        logger.info(f"Phoneme: {phoneme}")
        path_in = os.path.join(input_file, f"{phoneme}/{type_of}_siamese.npy")
        path_out = os.path.join(output_file, f"{phoneme}/pairwise_{type_of}_siamese.npy")
        vectors = np.load(path_in)

        logger.info(f"computing distance for {type_of}")
        distances = pdist(vectors, metric='cosine')

        logger.info("Saving pairwise distances")
        np.save(path_out, distances)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--phoneme_list", type=str)
    parser.add_argument("--type_of", type=str, help="type of vector, either phon or sem")
    args = parser.parse_args()

    compute_and_save_pairwise_distances(args.input_file, args.output_file, args.phoneme_list, args.type_of)
