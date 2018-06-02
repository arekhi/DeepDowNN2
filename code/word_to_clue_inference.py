import pickle 
import numpy as np
import helper_functions
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    import tensorflow as tf
    import keras

NUM_TRAIN = 1
WORD_IDX = 0
a_LSTM = 128

np.random.seed(0)
tf.set_random_seed(0)

# Read in word-clue pairs
with open('../data/word_clue_pairs.txt', 'rb') as fp:
    word_clue_pairs_list = pickle.load(fp)

# Read in word-glove pairs
with open('../data/word_glove_pairs.txt', 'rb') as fp:
    word_glove_pairs_dict = pickle.load(fp)
    word_to_index_dict = pickle.load(fp)
    index_to_word_dict = pickle.load(fp)
glove_length = len(word_glove_pairs_dict['a'])

# Make a new list: for the word-clue pairs whose words appear in the word-glove
#   dict, translate that pair into a pair [emb_word, emb_clue_list], where
#   emb_clue_list is the list [emb_clue_word_0, emb_clue_word_1, ...].
word_clue_embeddings_list, words, indices, clues, num_pairs_added, max_clue_length = helper_functions.choose_word_clue_pairs(NUM_TRAIN, word_clue_pairs_list, word_glove_pairs_dict, word_to_index_dict)

# Add start, end, and pad tokens to word-glove pairs dict and append start and end tokens to each clue (and pad)
word_glove_pairs_dict, word_to_index_dict, index_to_word_dict, training_clue_indices, clues = helper_functions.add_tokens(word_glove_pairs_dict, word_to_index_dict, index_to_word_dict, glove_length, clues, max_clue_length, np)

# Make the embedding matrix
embedding_matrix = np.zeros((len(word_glove_pairs_dict), glove_length))
for word in word_to_index_dict.keys():
    embedding_matrix[word_to_index_dict[word]] = np.array(word_glove_pairs_dict[word])

# Load the model
trained_model = keras.models.load_model('trained_model_experiment.h5')
encoder_layer_weights = trained_model.layers[5].get_weights()
decoder_layer_weights = trained_model.layers[6].get_weights()
dense_layer_weights = trained_model.layers[8].get_weights()

# Define the training model
embedding_layer = keras.layers.Embedding(len(word_glove_pairs_dict), glove_length, weights = [embedding_matrix], trainable = False, name = 'embedding')
encoder_LSTM = keras.layers.LSTM(a_LSTM, return_state = True, return_sequences = False, name = 'encoder_LSTM')
decoder_LSTM = keras.layers.LSTM(a_LSTM, return_state = True, return_sequences = True, name = 'decoder_LSTM') 
dropout_layer = keras.layers.Dropout(0.5)
dense_layer = keras.layers.TimeDistributed(keras.layers.Dense(len(word_glove_pairs_dict)))
softmax_activation = keras.layers.Activation('softmax')

a0 = keras.layers.Input(shape = (a_LSTM,), name = 'a0')
c0 = keras.layers.Input(shape = (a_LSTM,), name = 'c0')
word_index = keras.layers.Input(shape = (1,), dtype = 'int32', name = 'word_index')
clue_indices = keras.layers.Input(shape = (None,), dtype = 'int32', name = 'clue_indices')

x_word = embedding_layer(word_index)
encoder_output, a, c = encoder_LSTM(x_word, initial_state = [a0, c0])
x_clue = embedding_layer(clue_indices)
LSTM_output, _, _ = decoder_LSTM(x_clue, initial_state = [a, c])
output = dropout_layer(LSTM_output)
output = dense_layer(output)
output = softmax_activation(output)

model = keras.models.Model(inputs = [a0, c0, word_index, clue_indices], outputs = output)

# Define the inference set
NUM_INFER = 1
x_infer_a0 = np.zeros((NUM_INFER, a_LSTM))
x_infer_c0 = np.zeros((NUM_INFER, a_LSTM))
x_infer_word_index = np.array([indices[WORD_IDX]])
x_infer = [x_infer_a0, x_infer_c0, x_infer_word_index]

# Define the inference setup
encoder_model = keras.models.Model(inputs = [a0, c0, word_index], outputs = [a, c])
#keras.utils.plot_model(encoder_model, to_file='encoder_model.png', show_shapes = True)

clue_word_index = keras.layers.Input(shape = (None,), dtype = 'int32', name = 'clue_word_index')
x_clue = keras.layers.Embedding(len(word_glove_pairs_dict), glove_length, weights = [embedding_matrix], trainable = False, name = 'x_clue')(clue_word_index)

decoder_state_input_a = keras.layers.Input(shape = (a_LSTM,))
decoder_state_input_c = keras.layers.Input(shape = (a_LSTM,))
decoder_outputs, a, c = decoder_LSTM(x_clue, initial_state = [decoder_state_input_a, decoder_state_input_c])
decoder_outputs = dense_layer(decoder_outputs)
decoder_model = keras.models.Model(inputs = [clue_word_index] + [decoder_state_input_a, decoder_state_input_c], outputs = [decoder_outputs] + [a, c])
print(decoder_model.summary())
#keras.utils.plot_model(decoder_model, to_file='decoder_model.png', show_shapes = True)

states_inferred = encoder_model.predict(x_infer)
generated_clue = ['<START>']

stop_condition = False
max_length = 20
while not stop_condition:
    output_OH, a_infer, c_infer = decoder_model.predict([np.array([word_to_index_dict[generated_clue[-1]]])] + states_inferred)
    generated_word = index_to_word_dict[np.argmax(output_OH)]
    generated_clue.append(generated_word)
    if generated_word == '<END>' or len(generated_clue) == max_length:
        stop_condition = True
    states_inferred = [a_infer, c_infer]

print('\nWord: ' + index_to_word_dict[indices[WORD_IDX]])
print('\nActual clue: ' + ' '.join(word for word in clues[WORD_IDX]))
print('\nGenerated clue: ' + ' '.join(word for word in generated_clue) + '\n') 
