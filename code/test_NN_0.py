import pickle 
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    import keras

# Read in word-clue pairs
with open('../data/word_clue_pairs.txt', 'rb') as fp:
    word_clue_pairs_list = pickle.load(fp)

# Read in word-glove pairs
with open('../data/word_glove_pairs.txt', 'rb') as fp:
    word_glove_pairs_dict = pickle.load(fp)
glove_length = len(word_glove_pairs_dict['a'])

# Make a new list: for the word-clue pairs whose words appear in the word-glove
#   dict, translate that pair into a pair [emb_word, emb_clue_list], where
#   emb_clue_list is the list [emb_clue_word_0, emb_clue_word_1, ...].
NUM_TRAIN = 100
word_clue_embeddings_list = []
num_pairs_added = 0
done_flag = 0
words = []
clues = []
for pair in word_clue_pairs_list:
    if done_flag:
        break
    # end if
    word = pair[1].lower()
    if word in word_glove_pairs_dict:
        clue = pair[0].lower().split()
        clue_embeddings_list = []
        missing_word_flag = 0
        for clue_word in clue:
            if clue_word in word_glove_pairs_dict:
                clue_embeddings_list.append(word_glove_pairs_dict[clue_word])
            else: # if a word in the clue is not in the glove database, flag it and break
                missing_word_flag = 1
                break
                #clue_embeddings_list.append(glove_length * [0])
            # end if
        # end for
        if not missing_word_flag:
            words.append(word)
            word_embedding = word_glove_pairs_dict[word]
            clues.append(clue)
            word_clue_embeddings_list.append([word_embedding, clue_embeddings_list])
            num_pairs_added += 1
            if num_pairs_added >= NUM_TRAIN:
                done_flag = 1
            # end if
            if (num_pairs_added % 100000 ) == 0:
                print(num_pairs_added)
            # end if
        # end if
    # end if
# end for

print(num_pairs_added)

assert(len(words) == len(set(words)))

# Turn the training set into a matrix
training_matrix = np.zeros((NUM_TRAIN, glove_length))
for count, _ in enumerate(training_matrix):
    training_matrix[count] = word_clue_embeddings_list[count][0]
labels = (NUM_TRAIN // 2) * [0, 1]
assert(len(labels) == NUM_TRAIN)

# Define the model
model = keras.models.Sequential()
model.add(keras.layers.Embedding(NUM_TRAIN, glove_length, weights = [training_matrix], input_length = 1, trainable = False))
#model.add(keras.layers.Flatten())
#model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.LSTM(100))
model.add(keras.layers.Dense(1, activation = 'sigmoid'))

# Compile the model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])

# Summarize the model
print(model.summary())

# Fit the model
model.fit(list(range(NUM_TRAIN)), labels, epochs = 50, verbose = 0)

# Evaluate the model (against the training set for now)
loss, accuracy = model.evaluate(list(range(NUM_TRAIN)), labels, verbose = 0)
print('Accuracy: ' + str(100*accuracy) + '%')
