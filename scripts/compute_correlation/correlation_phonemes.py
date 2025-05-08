import os
import mantel
import logging
import argparse

import numpy as np
from csv import writer
from scipy.stats import spearmanr


logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def correlations(phoneme_list, base_path, output_csv):
    with open(phoneme_list, 'r', encoding="utf-8") as file:
        first_phonemes = [line.strip() for line in file.readlines()]


    for phoneme in first_phonemes:
        logger.info(f"Phoneme: {phoneme}")
        path_phon = os.path.join(base_path, f"{phoneme}/pairwise_phon_siamese.npy")
        path_sem = os.path.join(base_path, f"{phoneme}/pairwise_sem_siamese.npy")

        phon = np.load(path_phon)
        sem = np.load(path_sem)
        if len(phon) < 3:
            continue


        logger.info("Running Mantel test")
        mantel_test = mantel.test(phon, sem, method="spearman", perms=1000)
        significant = mantel_test.p < 0.05
        logger.info(f"Mantel test results: {mantel_test}, Significant: {significant}")

        if not os.path.exists(output_csv):
            with open(output_csv, "w") as file:
                writer_object = writer(file)
                writer_object.writerow(["Phoneme", "Corr", "Pvalue", "Significant"])

        with open(output_csv, 'a', newline='') as f:
            writer_object = writer(f)
            writer_object.writerow([phoneme, mantel_test.r, mantel_test.p, significant])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phoneme_list", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    correlations(args.phoneme_list, args.base_path, args.output_csv)

