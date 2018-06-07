import re
import string
import pickle

###############################################################################
# Store word-glove pairs
word_glove_pairs_dict_orig = {}
word_glove_pairs_dict = {}
word_to_index_dict = {}
index_to_word_dict = {}

NUM_DIM = 200
NUM_WORDS = 500000

glove_file = open('../../Downloads/glove.6B.' + str(NUM_DIM) + 'd.txt', 'r')

with open('../data/word_clue_pairs.txt', 'rb') as fp:
    word_clue_pairs_list = pickle.load(fp)


# num_gloves_keeped = 0
# done_flag = 0
#word = []
# indices = []
# clues = []
# word_present = 0
word_count = [] 
#same_clue = []

index = 0
for line in glove_file:
    # extract vocabulary word and its corresponding glove
    match_entry = re.search(r'(.*?) ', line) 
    if match_entry:
        entry = match_entry.group(1)
        glove = []
        for t in line.split()[1:]:
            try:
                glove.append(float(t))
            except ValueError:
                pass
        word_glove_pairs_dict_orig[entry] = glove
        # word_to_index_dict[entry] = index
        # index_to_word_dict[index] = entry
        index += 1
print(str(len(word_glove_pairs_dict_orig)) + ' entries in original gloves')
glove_file.close()

# print(word_glove_pairs_dict['hello'])
# assert(1==0)

for pair in word_clue_pairs_list:
    # extract word and clue without punctuation
    word = pair[1].lower()
    clue = pair[0].lower()
    for ch in [',', ':', '.', ';', '"', '\'', '!', '?', '$', '%', '(', ')']:
        clue = clue.replace(ch, '')
    clue = clue.replace('-', ' ')
    clue = clue.split() 
    clue.append(word)
    length = len(clue)
    # check duplicate and create a new dict stored only unique words in word-pair
    i = 1
    for word_in_clue in clue:
        if word_in_clue in word_glove_pairs_dict_orig:
            word_glove_pairs_dict[word_in_clue] = word_glove_pairs_dict_orig[word_in_clue]
            #if i == length:
            #    if word_in_clue not in word_count:
            #        word_count.append(word_in_clue)
        i += 1
    #        print(str(len(word_glove_pairs_dict)))
    #if clue not in same_clue:
    #    same_clue.append(clue)
# print(str(len(word_clue)) + ' different words in word-pair list')
# print(str(len(same_clue)) + 'different word/clue in word-pair list')
#print(str(len(word_count)) + ' unique words in the word data set')
# update index
index = 0
for key in word_glove_pairs_dict:
    word_to_index_dict[key] = index
    index_to_word_dict[index] = key
    index +=1

# Save all dictionaries 
with open('../data/word_glove_pairs_word_all.txt', 'wb') as fp:
    pickle.dump(word_glove_pairs_dict, fp)
    pickle.dump(word_to_index_dict, fp)
    pickle.dump(index_to_word_dict, fp)

# Print number of word-glove pairs
print(str(len(word_glove_pairs_dict)) + ' word/glove pairs were retrieved and stored.\n')

###############################################################################
