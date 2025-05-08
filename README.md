# A Cross-linguistic Exploration of the Correlation Between Form and Meaning: Examining the Role of Frequency, Phonesthemes, and Word Length
This is the repository for my master's thesis in Language Technology at the University of Oslo. 
The thesis investigates the correlation between phonetic embeddings and skip-gram embeddings across 19 languages.
 
The 19 languages looked at in this thesis are:
- Croatian
- Czech
- Dutch
- English
- German
- Hindi
- Hungarian
- Indonesian
- Italian
- Kazakh
- Polish
- Romanian
- Russian
- Spanish
- Swedish
- Ukrainian
- Urdu
- Uyghur
- Vietnamese

## Folder Structure
**scripts** contains all python scripts used in the thesis
 - **compute_correlation** contains the scripts used to compute pairwise distances between embeddings as well as scripts used to compute the correlation between the pairwise distances. There are one 'general' script for computing pairwise distances and correlations, as well as one script specifically for the different experiments.
 - **dataset** contains scripts used to combine the different conllu files into one file, as well as the scripts for lemmatisation for the languages where the conllu files did not contain lemmatised words, and the script to transcripe the words into IPA.
 - **onset_phoneme_clusters** contains scripts used to get the words for the second experiment of the thesis. The first is used to find all onset phoneme clusters and save a list containing all phoneme cluster as well as saving one word list for each onset phoneme cluster. The second script goes through the different word lists and connects the words to their corresponding vectors.
 - **training_model** contains the three training scripts for training the different embeddings models (two phonetic embedding models and one semantic embedding model). The Siamese Network Folder is taken directly from the [Phonetic Word Embedding Suite (PWESuite) Git Repo](https://github.com/zouharvi/pwesuite). There is also the get_vectors.py script which is used to save the vectors from the triplet margin model as a .npy file, as well as a python script which connects the different words in the word list to their corresponding vectors.
 - **word_length** contains the script used to get the words for the third experiment of the thesis. sort_based_on_length.py takes in a list of words and sorts them to different files based on their length, while get_vectors.py connects the words in the different files to their corresponding vectors.

**slurm** contains the slurm scripts used to run the scripts on the cluster
 - The folder structure inside the slurm folder follows the same structure as the scripts folder, with the same names. The slurm scripts are names according to the script they are running.

**word_lists** contains the word lists for each language used in the thesis. These are the words we are left with after training the skip-gram model and phonetic embeddings.



