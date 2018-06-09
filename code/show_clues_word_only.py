import pickle
import sys
import helper_functions

NUM_TRAIN = 1000000

# Read in word-clue pairs
with open('../data/word_clue_pairs.txt', 'rb') as fp:
    word_clue_pairs_list = pickle.load(fp)

# Read in word-glove pairs
with open('../data/word_glove_pairs_word_all.txt', 'rb') as fp:
    word_glove_pairs_dict = pickle.load(fp)
    word_to_index_dict = pickle.load(fp)
    index_to_word_dict = pickle.load(fp)
glove_length = len(word_glove_pairs_dict['a'])

# Make a new list: for the word-clue pairs whose words appear in the word-glove
#   dict, translate that pair into a pair [emb_word, emb_clue_list], where
#   emb_clue_list is the list [emb_clue_word_0, emb_clue_word_1, ...].
words, indices, clues, num_pairs_added, max_clue_length = helper_functions.choose_word_clue_pairs(NUM_TRAIN, word_clue_pairs_list, word_glove_pairs_dict, word_to_index_dict)

print(len(set(words)))

for i, word in enumerate(words):
    if word.lower() == sys.argv[1]:
        print(i, ' '.join(word for word in clues[i]))
