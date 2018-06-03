import re
import string
import pickle

###############################################################################
# Store word-glove pairs
word_glove_pairs_dict = {}
word_to_index_dict = {}
index_to_word_dict = {}
NUM_DIM = 50
NUM_WORDS = 20000

glove_file = open('../data/glove.6B.' + str(NUM_DIM) + 'd.txt', 'r')

common_file = open('../data/most_common_words_20k.txt', 'r')

common_words = []
for line in common_file:
    common_words.append(line.split()[0])
common_words = common_words[0:NUM_WORDS-1]
common_words.append('unk')

common_file.close()

index = 0
for line in glove_file:
    # extract vocabulary word and its corresponding glove
    match_entry = re.search(r'(.*?) ', line) 
    if match_entry:
        entry = match_entry.group(1)
        if entry in common_words:
            glove = []
            for t in line.split()[1:]:
                try:
                    glove.append(float(t))
                except ValueError:
                    pass
            word_glove_pairs_dict[entry] = glove
            word_to_index_dict[entry] = index
            index_to_word_dict[index] = entry
            index += 1

glove_file.close()

# Save all dictionaries 
with open('../data/word_glove_pairs.txt', 'wb') as fp:
    pickle.dump(word_glove_pairs_dict, fp)
    pickle.dump(word_to_index_dict, fp)
    pickle.dump(index_to_word_dict, fp)

# Print word-glove pairs
#for row in word_glove_pairs_dict:
#    print(row)

# Print number of word-glove pairs
print(str(len(word_glove_pairs_dict)) + ' word/glove pairs were retrieved and stored.\n')

###############################################################################
