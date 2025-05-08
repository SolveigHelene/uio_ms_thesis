import logging
import argparse

import numpy as np
from scipy.spatial.distance import pdist


logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_and_save_pairwise_distances(input_file, output_file):
    logger.info(f"Loading Vecotors for full pairwise")    
    vectors = np.load(input_file)

    logger.info("Computing pairwise cosine distances")
    distances = pdist(vectors, metric='cosine')

    np.save(output_file, distances)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file containing vectors")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file to save pairwise distances")
    args = parser.parse_args()

    compute_and_save_pairwise_distances(args.input_file, args.output_file)
