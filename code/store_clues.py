import csv
import re
import string
import pickle

###############################################################################
# Read in word-clue pairs
word_clue_pairs_csv = open('../data/word_clue_pairs.csv', newline = '')
reader = csv.reader(word_clue_pairs_csv, delimiter = ',')
word_clue_pairs_list = list()

for row in reader:
    word_clue_pairs_list.append(row)

# Sort list by word
word_clue_pairs_list.sort(key = lambda x: x[0])

with open('../data/word_clue_pairs.txt', 'wb') as fp:
    pickle.dump(word_clue_pairs_list, fp) 

# Close word-clue pairs file
word_clue_pairs_csv.close()

###############################################################################
