import os
import logging
import argparse

import numpy as np

logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
logger = logging.getLogger(__name__)

# get semantic and phoentic embeddings for the different words per beginning phoneme

def preprocess(file_path_phon_words, file_path_first_phonemes, file_path_sem_vectors, file_path_phon_vectors, lang, file_path_base):

    logger.info(f"Language running: {lang}")

    # file_path is the frequency list, and we want the words out
    logger.info("opening file with words for the phonetic embeddings")
    with open(file_path_phon_words, 'r') as f:
        words = [word.strip().replace(" ", "") for word in f.readlines()]
    
    logger.info("Opening file with the onset phonemes")
    with open(file_path_first_phonemes, 'r', encoding="utf-8") as file:
        first_phonemes = [line.strip() for line in file.readlines()]

    logger.info("loading phonetic and semantic embeddings")
    vectors_phon = np.load(file_path_phon_vectors)
    vectors_sem = np.load(file_path_sem_vectors)
    
    zipped_data = list(zip(words, vectors_phon, vectors_sem))

    for phoneme in first_phonemes:
        with open(os.path.join(file_path_base, f"{phoneme}/words.txt"), "r", encoding="utf-8") as word_file:
            found_words = [line.strip() for line in word_file.readlines() if line.startswith(phoneme)]
        logger.info(f"looking at phonene {phoneme}")
        # words, phon_vectors, sem_vectors = zip(*[(word, phon, sem) for word, phon, sem in zip(words, vectors_phon, vectors_sem) if word.startswith(phoneme)])
        filtered_data = [(word, phon, sem) for word, phon, sem in zipped_data if word.startswith(phoneme)]
        words_filtered, phon_vectors_filtered, sem_vectors_filtered = zip(*filtered_data)

        logger.info(f"how many words it should be: {len(found_words)}, len words that have been filtered: {len(words_filtered)}")
        assert len(found_words) == len(words_filtered)
        logger.info("Saving embeddings")
        np.save(os.path.join(file_path_base, f"{phoneme}/sem_siamese.npy"), sem_vectors_filtered)
        np.save(os.path.join(file_path_base, f"{phoneme}/phon_siamese.npy"), phon_vectors_filtered)
    
    logger.info(f"Done with {lang}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--file_path_phon_words", type=str, required=True, help="Path to the file with phonetic words")
    arg("--file_path_first_phonemes", type=str, required=True, help="Path to the file with list of all onset phoneme clusters")
    arg("--file_path_sem_vectors", type=str, required=True, help="Path to the file with semantic vectors")
    arg("--file_path_phon_vectors", type=str, required=True, help="Path to the file with phonetic vectors")
    arg("--lang", type=str, required=True, help="Language currently running")
    arg("--file_path_base", type=str, required=True, help="Base path for output files")
    args = parser.parse_args()

    preprocess(args.file_path_phon_words, args.file_path_first_phonemes, args.file_path_sem_vectors, args.file_path_phon_vectors, args.lang, args.file_path_base)
