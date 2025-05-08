import stanza
import logging
import argparse

from tqdm import tqdm
from smart_open import open


logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def lemmatize(input_file, output_file):
    nlp = stanza.Pipeline('id', processors='tokenize,lemma', verbose=False, lemma_batch_size=512)
    logger.info("loaded stanza for id")

    with open(input_file, 'rb', encoding="utf-8") as file:
        documents = [line.strip() for line in file]

    with open(output_file, "w", encoding="utf-8") as f:
        for doc in tqdm(documents, desc="Lemmatizing", unit="sentence"):
            stanza_doc = nlp(doc) 
            for sentence in stanza_doc.sentences:
                lemmatized_sentence = " ".join([word.lemma for word in sentence.words if word.lemma])
                f.write(lemmatized_sentence + "\n") 

    logger.info("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--lang_id", type=str, default="id")
    args = parser.parse_args()

    # Download model if not already downloaded
    #stanza.download(args.lang_id)

    lemmatize(args.input_file, args.output_file)