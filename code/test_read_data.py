import pickle

###############################################################################
# Read in word-clue pairs
with open('../data/word_clue_pairs.txt', 'rb') as fp:
    word_clue_pairs_list = pickle.load(fp)

print(len(word_clue_pairs_list))

###############################################################################

###############################################################################
# Read in word-defn pairs
with open('../data/word_defn_pairs.txt', 'rb') as fp:
    word_defn_pairs_list = pickle.load(fp)

print(len(word_defn_pairs_list))

###############################################################################

###############################################################################
# Read in word-glove pairs
with open('../data/word_glove_pairs.txt', 'rb') as fp:
    word_glove_pairs_dict = pickle.load(fp)

print(len(word_glove_pairs_dict))

###############################################################################
