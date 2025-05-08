import os
import mantel
import logging
import argparse

import numpy as np
from csv import writer
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform, cdist


logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
logger = logging.getLogger(__name__)



def main(file_path_sem, file_path_phon, lang, output_csv):
    logger.info(f"Language running: {lang}")


    logger.info("Loading phonetic pairwise distances")
    phon_dist = np.load(file_path_phon)


    logger.info("Loading semantic pairwise distances")
    sem_dist = np.load(file_path_sem)


    logger.info("Running Mantel test")
    mantel_test = mantel.test(phon_dist, sem_dist, method="spearman", perms=1000)
    significant = mantel_test.p < 0.05
    logger.info(f"Mantel test results: {mantel_test}, Significant: {significant}")

    if not os.path.exists(output_csv):
        with open(output_csv, "w") as file:
            writer_object = writer(file)
            writer_object.writerow(["Lang", "Corr", "PValue", "Significant"])

    with open(output_csv, 'a', newline='') as f:
        writer_object = writer(f)
        writer_object.writerow([lang, mantel_test.r, mantel_test.p, significant])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    parser.add_argument("--file_path_sem", type=str, required=True)
    parser.add_argument("--file_path_phon", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    args = parser.parse_args()

    main(args.file_path_sem, args.file_path_phon, args.lang, args.output_csv)
