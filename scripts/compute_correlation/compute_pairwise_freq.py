import logging
import argparse

import numpy as np
from scipy.spatial.distance import pdist, squareform


logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def main(word_freq_dir, sem_vec_dir, phon_vec_dir, output_dir, lang):

    logger.info(f"Language: {lang}")
    
    logger.info("Loading words and frequencies")
    with open(word_freq_dir, 'r') as f:
        lines = f.readlines()
        freqs = [float(line.strip().split()[1]) for line in lines]
        words = [line.strip().split()[0] for line in lines]
    
    logger.info("Loading vectors")

    sem_vec = np.load(sem_vec_dir)
    phon_vec = np.load(phon_vec_dir)

    logger.info("Finding percentiles")

    q5 = np.percentile(freqs, 15)    
    q25 = np.percentile(freqs, 30)  
    q75 = np.percentile(freqs, 70)  
    q95 = np.percentile(freqs, 85) 

    top_15 = []
    bottom_15 = []
    top_30 = []
    bottom_30 = []

    zipped = list(zip(words, freqs, sem_vec, phon_vec))

    logger.info("Going through zipped and saving to different percentiles")
    for word, freq, sem, phon in zipped:
        if freq > q95:
            top_15.append((word, freq, sem, phon)) 
        elif freq > q75:
            top_30.append((word, freq, sem, phon)) 
        elif freq < q5:
            bottom_15.append((word, freq, sem, phon)) 
        elif freq < q25:
            bottom_30.append((word, freq, sem, phon)) 
    
    logger.info("Computing pairwise cosine distances top and bottom 15%")

    top_15_sem_vectors = [sem for _, _, sem, _ in top_15]
    bottom_15_sem_vectors = [sem for _, _, sem, _ in bottom_15]
    
    top_15_phon_vectors = [phon for _, _, _, phon in top_15]
    bottom_15_phon_vectors = [phon for _, _, _, phon in bottom_15]

    distances_sem_top_15 = pdist(top_15_sem_vectors, metric='cosine')
    distances_sem_bottom_15 = pdist(bottom_15_sem_vectors, metric='cosine')

    distances_phon_top_15 = pdist(top_15_phon_vectors, metric='cosine')
    distances_phon_bottom_15 = pdist(bottom_15_phon_vectors, metric='cosine')

    np.save(f"{output_dir}/{lang}_top_15_sem", distances_sem_top_15)
    np.save(f"{output_dir}/{lang}_bottom_15_sem", distances_sem_bottom_15)
    np.save(f"{output_dir}/{lang}_top_15_phon", distances_phon_top_15)
    np.save(f"{output_dir}/{lang}_bottom_15_phon", distances_phon_bottom_15)

    logger.info("saved 1/2")

    del top_15_sem_vectors, bottom_15_sem_vectors, distances_sem_top_15, distances_sem_bottom_15

    logger.info("Computing pairwise cosine distances top and bottom 30%")

    top_30_sem_vectors = [sem for _, _, sem, _ in top_30]
    bottom_30_sem_vectors = [sem for _, _, sem, _ in bottom_30]
    
    top_30_phon_vectors = [phon for _, _, _, phon in top_30]
    bottom_30_phon_vectors = [phon for _, _, _, phon in bottom_30]

    distances_sem_top_30 = pdist(top_30_sem_vectors, metric='cosine')
    distances_sem_bottom_30 = pdist(bottom_30_sem_vectors, metric='cosine')

    distances_phon_top_30 = pdist(top_30_phon_vectors, metric='cosine')
    distances_phon_bottom_30 = pdist(bottom_30_phon_vectors, metric='cosine')

    np.save(f"{output_dir}/{lang}_top_30_sem", distances_sem_top_30)
    np.save(f"{output_dir}/{lang}_bottom_30_sem", distances_sem_bottom_30)
    np.save(f"{output_dir}/{lang}_top_30_phon", distances_phon_top_30)
    np.save(f"{output_dir}/{lang}_bottom_30_phon", distances_phon_bottom_30)

    logger.info("saved 2/2, Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_freq_dir", type=str, required=True, help="Path to the word frequency file")
    parser.add_argument("--sem_vec_dir", type=str, required=True, help="Path to the semantic vectors (NumPy array)")
    parser.add_argument("--phon_vec_dir", type=str, required=True, help="Path to the phonetic vectors (NumPy array)")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to directory where pairwise distances should be saved")
    parser.add_argument("--lang", type=str, required=True, help="Language looked at")

    args = parser.parse_args()

    main(args.word_freq_dir, args.sem_vec_dir, args.phon_vec_dir, args.output_dir, args.lang)