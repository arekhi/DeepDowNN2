import csv
import re
import string
import pickle

###############################################################################
# Store word-defn pairs from open-source dictionary in a txt file 
word_defn_pairs_dict = {}

# Go through the open-source dictionary, letter by letter
for letter in string.ascii_uppercase:
    print('Working on letter ' + letter + '\n')
    # Open and parse dictionary file for this letter
    gnuCIDE_file = open('../data/dictionary/gcide-0.52/CIDE.' + letter, 'r', errors = 'ignore')
    content = gnuCIDE_file.readlines()

#    print(content[0:10])
#    print(len(content))
#    print(range(len(content)))

    # If entry found in a line, store it and its definition (ignoring obsolete definitions)
    for i in range(len(content)):
#        print(i)
        line = content[i]
        match_word = re.search(r'<ent>(.*?)</ent>', line)
        if match_word:
            word = match_word.group(1).lower()
            keep_going = 1
            while keep_going and i < (len(content) - 1):
                i += 1
#                print(i)
                line = content[i]
                match_obs = re.search(r'Obs.', line)
                match_defn = re.search(r'<def>(.*?)</def>', line)
                match_word = re.search(r'<ent>(.*?)</ent>', line)
#                print(match_obs, match_defn, match_word)
                if match_word:
                    keep_going = 0
                    i -= 1
                    break
                if match_obs:
                    pass
                if (not match_obs) and match_defn:
#                    print('Defn found')
                    defn = match_defn.group(1).lower()
                    if word not in word_defn_pairs_dict:
                        word_defn_pairs_dict[word] = defn
                    keep_going = 0

    # Close dictionary file for this letter
    gnuCIDE_file.close()

# Print number of words in dictionary
print(str(len(word_defn_pairs_dict)) + ' word/definition pairs were retrieved and stored.\n')

# Save word-definition pairs dict
with open('../data/word_defn_pairs.txt', 'wb') as fp:
    pickle.dump(word_defn_pairs_dict, fp)
###############################################################################
