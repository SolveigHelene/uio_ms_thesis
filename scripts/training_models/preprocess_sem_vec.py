import logging
import argparse

import numpy as np

# https://www.w3.org/TR/elreq/#ethiopic_punctuation
amharic_symb = '፠፡።፣፤፥፦፧፨‘’“”‹›«»€…'
# https://en.wikipedia.org/wiki/Bengali_alphabet#Punctuation_marks, https://en.wikipedia.org/wiki/Bengali_numerals
bengali_symb = '০১২৩৪৫৬৭৮৯৹৷৶৴৵৸₹–।'
english_punc = r'!"#$%&\'\(\)*\+,-./:;<=>?@\[\\\]^_`{|}~'
other_punc = r'‌'
punctuations = english_punc + amharic_symb + bengali_symb

# Since numbers and special characters are removed with epitran, we have fewer phonetic embeddings compared to semantic embeddings
# This script filters out words and semantic embeddings that do not have a corresponding phonetic embedding

def main(file_path_sem, file_path_phon, file_path_vectors, lang):
    logging.basicConfig(
            format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
        )
    logger = logging.getLogger(__name__)

    logger.info(f"Language running: {lang}")

    # file_path is the frequency list, and we want the words out
    logger.info("opening file with words for the semantic embeddings")
    words_sem = []
    with open(file_path_sem, 'r') as f:
        lines = f.readlines()
        for line in lines:
            word = line.strip().split()[0]  # First element is the word, second is the frequency
            word = word.strip(punctuations) 
            words_sem.append(word)

    logger.info("opening file with words that have been transcribed to ipa")
    with open(file_path_phon, 'r') as file:
        lines = file.readlines()
        word_phon = [line.strip() for line in lines]

    logger.info("loading semantic embeddings")
    vectors_sem = np.load(file_path_vectors)
    logger.info(f"words_sem = {len(words_sem)}, vectors = {len(vectors_sem)}, word_phon = {len(word_phon)}")

    word_phon_set = set(word_phon)

    words_to_keep = []
    vectors_to_keep = []

    # only want semantic vectors for the words that appear in the phonetic vector set
    words_to_keep, vectors_to_keep = zip(*[(word, vec) for word, vec in zip(words_sem, vectors_sem) if word in word_phon_set])

    # logger.info(f"words_to_keep = {len(set(words_to_keep))}, word_phon = {len(set(word_phon))}")
    # logger.info(f"words_to_keep = {len(words_to_keep)}, vectors = {len(vectors_to_keep)}, word_phon = {len(word_phon)}")
    # logger.info(len(words_to_keep) == len(word_phon))
    assert len(words_to_keep) == len(word_phon)
    assert len(words_to_keep) == len(vectors_to_keep)
    

    # Save the filtered words and corresponding vectors
    logger.info("saving filtered list with words")
    with open(f'filtered_sem_emb/{lang}.txt', 'w') as f:
        for word in words_to_keep:
            f.write(word + '\n')

    logger.info("saving filtered list with vectors")
    np.save(f'filtered_sem_emb/{lang}.npy', vectors_to_keep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--file_path_sem", type=str, required=True, help="Path to the file with the words for the semantic embeddings")
    arg("--file_path_phon", type=str, required=True, help="Path to the file with the words that have been transcribed to ipa")
    arg("--file_path_vectors", type=str, required=True, help="Path to the file with the semantic embeddings")
    arg("--lang", type=str, required=True, help="Language currently running")
    args = parser.parse_args()

    main(args.file_path_sem, args.file_path_phon, args.file_path_vectors, args.lang)