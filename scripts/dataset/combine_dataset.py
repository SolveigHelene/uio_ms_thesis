import os
import lzma
import gzip
import logging
import argparse
import multiprocessing

from conllu import parse_incr
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_file(file_path, stop_words):
    """Read content from an .xz file, parse it, and process sentences."""
    logging.info(f"Processing file: {file_path}")
    sentences = []

    with lzma.open(file_path, 'rt', encoding='utf-8') as f:
        logging.info("**********************")
        logging.info(f"Parsing {file_path}")
        logging.info("**********************")
        
        for sentence in parse_incr(f):  # parse_incr yields sentences one by one
            sentence_lemmas = [
                token['lemma'].lower()
                for token in sentence
                if token['upos'] not in stop_words
            ]
            if sentence_lemmas:
                sentences.append(' '.join(sentence_lemmas))
        logging.info("**********************")
        logging.info(f"Done with Parsing {file_path}")
        logging.info("**********************")

    return sentences

def compress_to_gz(file_path, compressed):
    with open(file_path, 'r') as f_in: 
        with gzip.open(compressed, 'wb') as f_out:  
            f_out.write(f_in.read())  



def main(input_dir, output_dir, cores):
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    xz_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.xz')]

    if not xz_files:
        logger.error(f"No .xz files found in the directory: {input_dir}")
        exit(1)

    logger.info(f"Found {len(xz_files)} .xz files in {input_dir}. Combining them...")

    if cores:
        cores = int(cores)
    else:
        cores = (
            multiprocessing.cpu_count() - 1
        )
    logger.info(f"Number of cores to use: {cores}")

    # connllu stop_words abbrevations
    stop_words = ["DET", "PROPN", "CCONJ", "PUNCT", "PART", "ADP", "SCONJ", "PRON"]

  
    with open(output_dir, 'w', encoding='utf-8') as out_f:
            
        with ProcessPoolExecutor(max_workers=cores) as executor:
            futures = {executor.submit(process_file, file, stop_words): file for file in xz_files}
            
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    sentences = future.result()
                    if sentences:
                        out_f.write("\n".join(sentences))
                except Exception as e:
                    logging.error(f"Error processing file {file_path}: {e}")

        
    logger.info(f"Combined content saved to: {output_dir}")

    compress_to_gz(output_dir, output_dir.replace(".txt", ".txt.gz"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--input_dir", help="Path to a training corpus", required=True)
    arg("--output_dir", help="Path to where output should be saved", required=True)
    arg("--cores", default=False, help="Limit on the number of cores to use")

    args = parser.parse_args()
    input_dir=args.input_dir
    output_dir=args.output_dir
    cores=args.cores

    main(input_dir, output_dir, cores)




    

    