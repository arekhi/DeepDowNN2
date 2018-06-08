import pickle 
import numpy as np
import helper_functions
import warnings
import sys
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    import tensorflow as tf
    import keras

NUM_TRAIN = 50000 #2**14
MAX_DEFN_LEN = 20
WORD_IDX = int(sys.argv[1])
a_LSTM = 128

np.random.seed(0)
tf.set_random_seed(0)

# Read in word-clue pairs
with open('../data/word_clue_pairs.txt', 'rb') as fp:
    word_clue_pairs_list = pickle.load(fp)

# Read in word-glove pairs
with open('../data/word_glove_pairs_word_all.txt', 'rb') as fp:
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

print('\nNum pairs added: ' + str(num_pairs_added) + '\n')

# Add start, end, and pad tokens to word-glove pairs dict, clip definitions, and append start, end, and pad tokens to each clue and definition
word_glove_pairs_dict, word_to_index_dict, index_to_word_dict, training_clue_indices, definition_indices, clues, definitions = helper_functions.add_tokens_with_dict(word_glove_pairs_dict, word_to_index_dict, index_to_word_dict, glove_length, clues, max_clue_length, np, definitions, MAX_DEFN_LEN)

# Make the embedding matrix
embedding_matrix = np.zeros((len(word_glove_pairs_dict), glove_length))
for word in word_to_index_dict.keys():
    embedding_matrix[word_to_index_dict[word]] = np.array(word_glove_pairs_dict[word])

# Load the model
trained_model = keras.models.load_model('trained_model_with_defn.h5', custom_objects = {'keras': keras, 't': 0})
#print(trained_model.summary())

encoder_layer_weights = trained_model.layers[7].get_weights()
encoder_bwd_layer_weights = trained_model.layers[8].get_weights()
dense_a_weights = trained_model.layers[12].get_weights()
dense_c_weights = trained_model.layers[13].get_weights()
decoder_layer_weights = trained_model.layers[15].get_weights()
dense_encoder_weights = trained_model.layers[22].get_weights()
attn_dense_1_weights = trained_model.layers[27].get_weights()
attn_dense_2_weights = trained_model.layers[33].get_weights()
attn_dense_3_weights = trained_model.layers[46].get_weights()
final_dense_weights = trained_model.layers[54].get_weights()

# Define the training model
masking_layer = keras.layers.Masking(mask_value = word_to_index_dict['<PAD>'], input_shape = (None,))
embedding_layer = keras.layers.Embedding(len(word_glove_pairs_dict), glove_length, weights = [embedding_matrix], trainable = False, name = 'embedding')
encoder_LSTM = keras.layers.LSTM(a_LSTM, return_state = True, return_sequences = True, name = 'encoder_LSTM', weights = encoder_layer_weights, recurrent_dropout = 0.4)
encoder_LSTM_bwd = keras.layers.LSTM(a_LSTM, return_state = True, return_sequences = True, name = 'encoder_LSTM_bwd', go_backwards = True, weights = encoder_bwd_layer_weights, recurrent_dropout = 0.2)
dense_encoder_output = keras.layers.Dense(a_LSTM, activation = 'tanh', weights = dense_encoder_weights) 
dense_between_a = keras.layers.Dense(a_LSTM, activation = 'tanh', weights = dense_a_weights)
dense_between_c = keras.layers.Dense(a_LSTM, activation = 'tanh', weights = dense_c_weights)
decoder_LSTM = keras.layers.LSTM(a_LSTM, return_state = True, return_sequences = True, name = 'decoder_LSTM', weights = decoder_layer_weights, recurrent_dropout = 0.2)
squeezer = keras.layers.Lambda(lambda x: x[:, 0, :])
repeater = keras.layers.RepeatVector(MAX_DEFN_LEN)
attn_dense_1 = keras.layers.Dense(64, activation = "tanh", weights = attn_dense_1_weights) 
attn_dropout = keras.layers.Dropout(0.2)
attn_dense_2 = keras.layers.Dense(1, activation = "relu", weights = attn_dense_2_weights)
attn_softmax = keras.layers.Softmax(axis = 1)
attn_dot = keras.layers.Dot(axes = 1)
attn_dense_3 = keras.layers.Dense(64, weights = attn_dense_3_weights) 
dropout_layer = keras.layers.Dropout(0.4)
dense_layer = keras.layers.Dense(len(word_glove_pairs_dict), weights = final_dense_weights)
softmax_activation = keras.layers.Activation('softmax')

a0 = keras.layers.Input(shape = (a_LSTM,), name = 'a0')
c0 = keras.layers.Input(shape = (a_LSTM,), name = 'c0')
defn_indices = keras.layers.Input(shape = (MAX_DEFN_LEN,), dtype = 'int32', name = 'defn_indices')
clue_indices = keras.layers.Input(shape = (None,), dtype = 'int32', name = 'clue_indices')

masked_defn_indices = masking_layer(defn_indices)
x_defn = embedding_layer(masked_defn_indices)
encoder_output, a, c = encoder_LSTM(x_defn, initial_state = [a0, c0])
encoder_bwd_output, a_bwd, c_bwd = encoder_LSTM_bwd(x_defn, initial_state = [a0, c0])
encoder_output_concat = keras.layers.Concatenate()([encoder_output, encoder_bwd_output])
encoder_output_densed = dense_encoder_output(encoder_output_concat)
a_concat = keras.layers.Concatenate()([a, a_bwd])
c_concat = keras.layers.Concatenate()([c, c_bwd])
a_passed_enc = dense_between_a(a_concat)
c_passed_enc = dense_between_c(c_concat)
a_passed = a_passed_enc
c_passed = c_passed_enc

for t in range(max_clue_length + 2):
    clue_index = keras.layers.Lambda(lambda x: keras.backend.expand_dims(x[:, t], axis = -1))(clue_indices) 
    masked_clue_index = masking_layer(clue_index)
    x_clue = embedding_layer(masked_clue_index)
    output_dec, a_passed, c_passed = decoder_LSTM(x_clue, initial_state = [a_passed, c_passed])
    output = squeezer(output_dec)
    output = repeater(output)
    output = keras.layers.Concatenate(axis = -1)([output, encoder_output_densed])
    output = attn_dense_1(output)
    output = attn_dropout(output)
    output = attn_dense_2(output)
    output = attn_softmax(output)
    output = attn_dot([output, encoder_output_densed])
    output = keras.layers.Concatenate()([output, output_dec])
    output = attn_dense_3(output)
    output = dropout_layer(output)
    output = dense_layer(output)
    output = softmax_activation(output)
    if t == 0:
        outputs = output
    else:
        outputs = keras.layers.Concatenate(axis = 1)([outputs, output])

model = keras.models.Model(inputs = [a0, c0, defn_indices, clue_indices], outputs = outputs)

# Define the inference set
NUM_INFER = 1
x_infer_a0 = np.zeros((NUM_INFER, a_LSTM))
x_infer_c0 = np.zeros((NUM_INFER, a_LSTM))
x_infer_defn_indices = np.array([definition_indices[WORD_IDX]])
x_infer = [x_infer_a0, x_infer_c0, x_infer_defn_indices]

# Define the inference setup
encoder_model = keras.models.Model(inputs = [a0, c0, defn_indices], outputs = [a_passed_enc, c_passed_enc] + [encoder_output_densed])
#print(encoder_model.summary())
#keras.utils.plot_model(encoder_model, to_file='encoder_model.png', show_shapes = True)

clue_word_index = keras.layers.Input(shape = (1,), dtype = 'int32', name = 'clue_word_index')

decoder_state_input_a = keras.layers.Input(shape = (a_LSTM,))
decoder_state_input_c = keras.layers.Input(shape = (a_LSTM,))
decoder_attn_input = keras.layers.Input(shape = (MAX_DEFN_LEN, a_LSTM))

dec_a_passed = decoder_state_input_a
dec_c_passed = decoder_state_input_c

dec_masked_clue_index = masking_layer(clue_word_index)
dec_x_clue = embedding_layer(dec_masked_clue_index)
dec_output_LSTM, dec_a_passed, dec_c_passed = decoder_LSTM(dec_x_clue, initial_state = [dec_a_passed, dec_c_passed])
dec_output = squeezer(dec_output_LSTM)
dec_output = repeater(dec_output)
dec_output = keras.layers.Concatenate(axis = -1)([dec_output, decoder_attn_input])
dec_output = attn_dense_1(dec_output)
dec_output = attn_dropout(dec_output)
dec_output = attn_dense_2(dec_output)
dec_output = attn_softmax(dec_output)
dec_output = attn_dot([dec_output, decoder_attn_input])
dec_output = keras.layers.Concatenate()([dec_output, dec_output_LSTM])
dec_output = attn_dense_3(dec_output)
dec_output = dropout_layer(dec_output)
dec_output = dense_layer(dec_output)
dec_output = softmax_activation(dec_output)

decoder_model = keras.models.Model(inputs = [clue_word_index] + [decoder_state_input_a, decoder_state_input_c] + [decoder_attn_input], outputs = [dec_output] + [dec_a_passed, dec_c_passed])
#print(decoder_model.summary())
#keras.utils.plot_model(decoder_model, to_file='decoder_model.png', show_shapes = True)

print('\nWord: ' + index_to_word_dict[indices[WORD_IDX]])
print('\nDefinition: ' + ' '.join(word for word in definitions[WORD_IDX]))
print('\nActual clue: ' + ' '.join(word for word in clues[WORD_IDX]) + '\n')

encoder_a_output, encoder_c_output, encoder_attn_output = encoder_model.predict(x_infer)
generated_clue = ['<START>']

stop_condition = False
max_length = 20
for i in range(10):
    while not stop_condition:
        output_probs, a_infer, c_infer = decoder_model.predict([np.array([word_to_index_dict[generated_clue[-1]]])] + [encoder_a_output, encoder_c_output, encoder_attn_output])
        print(np.sort(output_probs.ravel())[-5:])
    #    print(np.sum(output_probs.ravel()))
        next_idx = np.random.choice(len(word_to_index_dict), p = output_probs.ravel())
    #    next_idx = np.argmax(output_probs) # Choose the most probable next word
        generated_word = index_to_word_dict[next_idx] # Sample the next word according to the outputted probabilities
        generated_clue.append(generated_word)
        if generated_word == '<END>' or len(generated_clue) == max_length:
            stop_condition = True
        encoder_a_output = a_infer
        encoder_c_output = c_infer
    print('\nGenerated clue: ' + ' '.join(word for word in generated_clue) + '\n')
    stop_condition = False
    generated_clue = ['<START>']
    encoder_a_output, encoder_c_output, _ = encoder_model.predict(x_infer)
