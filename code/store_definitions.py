import csv
import re
import string
import pickle

###############################################################################
# Store word-defn pairs from open-source dictionary in a csv file 
word_defn_pairs_list = list()

# Go through the open-source dictionary, letter by letter
for letter in string.ascii_uppercase:
    print('Working on letter ' + letter + '\n')
    # Open and parse dictionary file for this letter
    gnuCIDE_file = open('../data/dictionary/gcide-0.52/CIDE.' + letter, 'r', errors = 'ignore')

    # If entry found in a line, store it and its definition
    for line in gnuCIDE_file:
        match_word = re.search(r'<ent>(.*?)</ent>', line)
        if match_word:
            word = match_word.group(1)
            line = next(gnuCIDE_file)
            match_defn = re.search(r'<def>(.*?)</def>', line)
            if match_defn:
                defn = match_defn.group(1)
                word_defn_pairs_list.append([word,defn])

    # Close dictionary file for this letter
    gnuCIDE_file.close()

# Print number of words in dictionary
print(str(len(word_defn_pairs_list)) + ' word/definition pairs were retrieved and stored.\n')

# Save word-definition pairs list
with open('../data/word_defn_pairs.txt', 'wb') as fp:
    pickle.dump(word_defn_pairs_list, fp)

# Print word-definition pairs
#for row in word_defn_pairs_list:
#    print(row[0])

###############################################################################
