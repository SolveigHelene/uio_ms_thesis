
import tqdm
import epitran
import panphon
import argparse

from emoji import is_emoji


# https://www.w3.org/TR/elreq/#ethiopic_punctuation
amharic_symb = '፠፡።፣፤፥፦፧፨‘’“”‹›«»€…'
# https://en.wikipedia.org/wiki/Bengali_alphabet#Punctuation_marks, https://en.wikipedia.org/wiki/Bengali_numerals
bengali_symb = '০১২৩৪৫৬৭৮৯৹৷৶৴৵৸₹–।'
english_punc = r'!"#$%&\'\(\)*\+,-./:;<=>?@\[\\\]^_`{|}~'
other_punc = r'‌'
punctuations = english_punc + amharic_symb + bengali_symb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--input_file", type=str, required=True)
    arg("--output_file", type=str, required=True)
    arg("--epitran_code", type=str, required=True)
    args = parser.parse_args()

    ft = panphon.FeatureTable()
    epi = epitran.Epitran(args.epitran_code)

    tokens_ipa = []
    with open(args.input_file, 'r', encoding='utf-8') as f, \
        open(args.output_file, 'w', encoding='utf-8') as f_out:

        for line in tqdm.tqdm(f):
            word = line.strip()  # Decode and strip whitespace/newlines
            word = word.strip(punctuations)  # Strip punctuation around the word
            if not word or any(c in "0123456789" or is_emoji(c) for c in word):
                continue

            segments = ft.ipa_segs(epi.transliterate(word))
            if not segments:
                continue
            f_out.write(' '.join(segments) + '\n')