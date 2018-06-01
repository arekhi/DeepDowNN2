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

# Print clues
#for row in word_clue_pairs_list: 
#    print(row[1])

with open('../data/word_clue_pairs.txt', 'wb') as fp:
    pickle.dump(word_clue_pairs_list, fp) 

# Close word-clue pairs file
word_clue_pairs_csv.close()

###############################################################################
