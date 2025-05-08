import math
import logging
import argparse
import multiprocessing

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# Setting up logging
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

def vectorize_chunk(chunk, vectorizer_type, vectorizer_args):
    """Vectorize a chunk of data."""
    if vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(**vectorizer_args)
    elif vectorizer_type == "count":
        vectorizer = CountVectorizer(**vectorizer_args)
    else:
        raise ValueError("Unsupported vectorizer type.")

    return vectorizer.fit_transform(chunk)

def transform_chunk(chunk, vectorizer):
    """Transform a chunk using the fitted vectorizer."""
    return vectorizer.transform(chunk)

def main(vectorizer_type, dataset):


    num_workers = multiprocessing.cpu_count() - 1
    logger.info(f"Number of cores to use: {num_workers}")

    # Read dataset
    with open(dataset, "r") as f:
        data_local = f.read().strip().split("\n")
    logger.info("Dataset loaded.")

    vectorizer_args = {
        "max_features": 1024,
        "ngram_range": (1, 3),
        "stop_words": None,
        "analyzer": "char",
        "min_df": 0.0,
    }

    # Fit the vectorizer on the entire dataset first
    logger.info("Fitting vectorizer on the entire dataset.")
    if vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(**vectorizer_args)
    elif vectorizer_type == "count":
        vectorizer = CountVectorizer(**vectorizer_args)
    else:
        raise ValueError("Unsupported vectorizer type.")

    vectorizer.fit(data_local)  # Fit on the full dataset
    logger.info("Vectorizer fitted.")
    

    # Split data into chunks for parallel processing, making sure one chunk isn't much smaller
    chunk_size = math.ceil(len(data_local) / num_workers)
    data_chunks = [data_local[i:i + chunk_size] for i in range(0, len(data_local), chunk_size)]
    logger.info(f"Data split into {len(data_chunks)} chunks.")
    
    logger.info("Starting vectorization.")
    vectorized_chunks = [vectorizer.transform(chunk) for chunk in data_chunks]

    logger.info("Vectorization complete.")

    logger.info("Concatenating final results.")
    data_final = np.vstack([chunk.toarray() for chunk in vectorized_chunks])

    #PCA
    logger.info("Fitting scaler and PCA.")
    pca = PCA(n_components=300)
    pca.fit(data_final)

    logger.info("Transforming data")
    final = pca.transform(data_final)

    return final


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--vectorizer", default="tfidf", help="Type of vectorizer to use: tfidf or count")
    args.add_argument("--lang", type=str, required=True, help="Language currently running")
    args.add_argument("--dataset", type=str, required=True, help="Path to the dataset file")
    args = args.parse_args()
    logger.info(f"Language {args.lang}")

    data_out = main(args.vectorizer, args.dataset)

    print("Saving vectors")
    np.save(f"phon_count/{args.lang}_count_{args.vectorizer}.npy", data_out)
    print("done")