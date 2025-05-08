
import gensim
import logging
import argparse
import multiprocessing

from os import path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--corpus", help="Path to a training corpus (can be compressed)", required=True)
    arg("--lang", type=str, help="ISO code for the language the model is to be trained for", required=True)
    arg("--cores", default=False, help="Limit on the number of cores to use")
    arg("--sg", default=0, type=int, help="Use Skipgram (1) or CBOW (0)")
    arg("--window", default=10, type=int, help="Size of context window")
    arg("--max_final_vocab", default=200_000, type=int, help="Max words in final vocab")
    arg("--epochs", default=4, type=int, help="How many epochs to train the model")
    arg("--negative", default=5, type=int, help="How many negative samples")
    arg("--threshold", default=0.0001, type=int, help="Loss threshold for when to stop training")
    arg("--no_improv", default=3, type=int, help="Max number of steps without change in loss during training")
    
    args = parser.parse_args()

    # Setting up logging:
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    # This will be our training corpus to infer word embeddings from.
    # Most probably, a gzipped text file, one doc/sentence per line:
    corpus = args.corpus
    
    #with lzma.open(corpus, 'rt', encoding='utf-8') as f:
    f = open(corpus, 'r', encoding='utf-8')
    data = gensim.models.word2vec.LineSentence(f)

    
    # Train the Word2Vec model using the preprocessed sentences

    # How many workers (CPU cores) to use during the training?
    if args.cores:
        # Use the number of cores we are told to use (in a SLURM file, for example):
        cores = int(args.cores)
    else:
        # Use all cores we have access to except one
        cores = (
            multiprocessing.cpu_count() - 1
        )
    logger.info(f"Number of cores to use: {cores}")

    # Setting up training hyperparameters:
    # Use Skipgram (1) or CBOW (0) algorithm?
    skipgram = args.sg
    # Context window size (e.g., 2 words to the right and to the left)
    window = args.window

    vectorsize = 300  # Dimensionality of the resulting word embeddings.

    # For how many epochs to train a model (how many passes over corpus)?
    epochs = args.epochs
    negative_samples = args.negative
    max_final_vocab = args.max_final_vocab
    lang = args.lang

    # Start actual training!
    logger.info("Starts training")
    model = gensim.models.Word2Vec(
        data,
        vector_size=vectorsize,
        window=window,
        workers=cores,
        sg=skipgram,
        negative=negative_samples,
        max_final_vocab=max_final_vocab,
        epochs=epochs,
        sample=0.001,
    )

    # Saving the resulting model to a file
    filename = f"{lang}_skipgram_" + path.basename(corpus).replace(".txt.xz", ".model")
    logger.info(filename)

    # Save the model without the output vectors (what you most probably want):
    model.wv.save(filename)
    logger.info(f"Words kept: {len(model.wv.index_to_key)}")

    # Save word frequency list
    def save_frequency_list(model, output_path):
        logger.info("Saving frequency list.")
        with open(output_path, 'w', encoding='utf-8') as f:
            word_frequencies = {word: model.wv.get_vecattr(word, 'count') for word in model.wv.key_to_index}

            # Sort by frequency (descending)
            sorted_frequencies = sorted(word_frequencies.items(), key=lambda item: item[1], reverse=True)
            for word, freq in sorted_frequencies:
                f.write(f"{word}\t{freq}\n")
        logger.info(f"Frequency list saved to {output_path}")

    freq_filename = f"frequency_{lang}_" + path.basename(corpus).replace(".txt.xz", ".txt")
    save_frequency_list(model, freq_filename)
    f.close()
    # model.save(filename)  # If you intend to train the model further