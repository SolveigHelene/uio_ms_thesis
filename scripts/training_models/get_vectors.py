import argparse
import logging

import numpy as np

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Script used for getting out the vectors from the .txt file that is created when training siamese network

def main(input, output):
    vectors = []    

    logger.info("Starting going through the file")
    with open(input, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            vector = list(map(float, parts[-300:]))  # Take the last 300 elements
            vectors.append(vector)
    
    logger.info("converting to numpy array and saving")
    vector_array = np.array(vectors)
    np.save(output, vector_array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    
    main(args.input_file, args.output_file)
