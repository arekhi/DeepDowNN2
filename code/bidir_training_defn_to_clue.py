import pickle 
import numpy as np
import helper_functions
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    import tensorflow as tf
    import keras

np.random.seed(0)
tf.set_random_seed(0)

NUM_TRAIN = 2**15
MAX_DEFN_LEN = 20
FRAC_VAL = 0.01
NUM_EPOCH = 3
a_LSTM = 128

# Read in word-clue pairs
with open('../data/word_clue_pairs.txt', 'rb') as fp:
    word_clue_pairs_list = pickle.load(fp)

# Read in word-glove pairs
with open('../data/word_glove_pairs.txt', 'rb') as fp:
    word_glove_pairs_dict = pickle.load(fp)
    word_to_index_dict = pickle.load(fp)
    index_to_word_dict = pickle.load(fp)
glove_length = len(word_glove_pairs_dict['a'])

# Read in word-definition pairs
with open('../data/word_defn_pairs.txt', 'rb') as fp:
    word_defn_pairs_dict = pickle.load(fp)

# Make a new list: for the word-clue pairs whose words appear in the word-glove
#   dict and in the GCIDE dict, translate that pair into a pair [emb_word, 
#   emb_clue_list], where emb_clue_list is the list [emb_clue_word_0, 
#   emb_clue_word_1, ...].
words, indices, clues, definitions, num_pairs_added, max_clue_length, max_defn_length = helper_functions.choose_word_clue_pairs_with_dict(NUM_TRAIN, word_clue_pairs_list, word_glove_pairs_dict, word_to_index_dict, word_defn_pairs_dict)

print(num_pairs_added)
#for i in range(20):
#    print(words[i], definitions[i])

# Add start, end, and pad tokens to word-glove pairs dict, clip definitions, and append start, end, and pad tokens to each clue and definition
word_glove_pairs_dict, word_to_index_dict, index_to_word_dict, training_clue_indices, definition_indices, clues, definitions = helper_functions.add_tokens_with_dict(word_glove_pairs_dict, word_to_index_dict, index_to_word_dict, glove_length, clues, max_clue_length, np, definitions, MAX_DEFN_LEN)

# Define the training set
x_train_a0 = np.zeros((num_pairs_added, a_LSTM))
x_train_c0 = np.zeros((num_pairs_added, a_LSTM))
# x_train_word_index = np.array(indices)
x_train_definition_indices = np.array(definition_indices) 
x_train_clue_indices = np.array(training_clue_indices)
x_train = [x_train_a0, x_train_c0, x_train_definition_indices, x_train_clue_indices]

#print(max_clue_length, len(word_glove_pairs_dict))

y_train = np.zeros((num_pairs_added, max_clue_length + 2, len(word_glove_pairs_dict)), dtype = 'float16')
for m in range(num_pairs_added):
    clue = clues[m]
    shifted_clue = clue[1:] + ['<PAD>']
    for i in range(max_clue_length + 2):
        y_train[m, i, word_to_index_dict[shifted_clue[i]]] = 1

# Make the embedding matrix
embedding_matrix = np.zeros((len(word_glove_pairs_dict), glove_length))
for word in word_to_index_dict.keys():
    embedding_matrix[word_to_index_dict[word]] = np.array(word_glove_pairs_dict[word])

# Define the training model
masking_layer = keras.layers.Masking(mask_value = word_to_index_dict['<PAD>'], input_shape = (None,))
embedding_layer = keras.layers.Embedding(len(word_glove_pairs_dict), glove_length, weights = [embedding_matrix], trainable = False, name = 'embedding')
encoder_LSTM = keras.layers.LSTM(a_LSTM, return_state = True, return_sequences = False, name = 'encoder_LSTM')
encoder_LSTM_bwd = keras.layers.LSTM(a_LSTM, return_state = True, return_sequences = False, name = 'encoder_LSTM_bwd', go_backwards = True)
dense_between_a = keras.layers.Dense(a_LSTM, activation = 'tanh')
dense_between_c = keras.layers.Dense(a_LSTM, activation = 'tanh')
decoder_LSTM = keras.layers.LSTM(a_LSTM, return_state = True, return_sequences = True, name = 'decoder_LSTM')
dropout_layer = keras.layers.Dropout(0.5)
dense_layer = keras.layers.TimeDistributed(keras.layers.Dense(len(word_glove_pairs_dict)))
softmax_activation = keras.layers.Activation('softmax')

a0 = keras.layers.Input(shape = (a_LSTM,), name = 'a0')
c0 = keras.layers.Input(shape = (a_LSTM,), name = 'c0')
defn_indices = keras.layers.Input(shape = (None,), dtype = 'int32', name = 'defn_indices')
clue_indices = keras.layers.Input(shape = (None,), dtype = 'int32', name = 'clue_indices')

#masked_defn_indices = masking_layer(defn_indices)
x_defn = embedding_layer(defn_indices)
encoder_output, a, c = encoder_LSTM(x_defn, initial_state = [a0, c0])
encoder_bwd_output, a_bwd, c_bwd = encoder_LSTM_bwd(x_defn, initial_state = [a0, c0])
a_concat = keras.layers.Concatenate()([a, a_bwd])
c_concat = keras.layers.Concatenate()([c, c_bwd])
a_passed = dense_between_a(a_concat)
c_passed = dense_between_c(c_concat)

masked_clue_indices = masking_layer(clue_indices)
x_clue = embedding_layer(masked_clue_indices)
output, _, _ = decoder_LSTM(x_clue, initial_state = [a_passed, c_passed])
output = dropout_layer(output)
output = dense_layer(output)
output = softmax_activation(output)

model = keras.models.Model(inputs = [a0, c0, defn_indices, clue_indices], outputs = output)

# Compile the training model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy']) 

# Summarize the training model
print(model.summary())

# Visualize training model
#keras.utils.plot_model(model, to_file='model.png', show_shapes = True)

# Fit the training model (train)
hist = model.fit(x_train, y_train, validation_split = FRAC_VAL, epochs = NUM_EPOCH, verbose = 1)
with open('model_stats.txt', 'wb') as fp: 
    pickle.dump(hist.history, fp)

model.save('trained_model_with_defn.h5')
