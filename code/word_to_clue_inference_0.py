import pickle 
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    import tensorflow as tf
    import keras

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
NUM_TRAIN = 1000000 
word_clue_embeddings_list = []
num_pairs_added = 0
done_flag = 0
words = []
indices = []
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
            indices.append(word_to_index_dict[word])
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

max_clue_length = 0
for clue in clues:
    if len(clue) > max_clue_length:
        max_clue_length = len(clue)

# Add start, end, and pad tokens to word-glove pairs dict and append start and end tokens to each clue (and pad)
#np.random.seed(0)
start_token = np.random.randn(glove_length,)
end_token = np.random.randn(glove_length,)
pad_token = np.zeros((glove_length,))
word_glove_pairs_dict['<START>'] = start_token
word_glove_pairs_dict['<END>'] = end_token
word_glove_pairs_dict['<PAD>'] = pad_token
word_to_index_dict['<START>'] = len(word_to_index_dict)
word_to_index_dict['<END>'] = len(word_to_index_dict)
word_to_index_dict['<PAD>'] = len(word_to_index_dict) 
index_to_word_dict[word_to_index_dict['<START>']] = '<START>'
index_to_word_dict[word_to_index_dict['<END>']] = '<END>'
index_to_word_dict[word_to_index_dict['<PAD>']] = '<PAD>'

training_clue_indices = []
for i in range(len(clues)):
    clue = ['<START>'] + clues[i] + ['<END>']
    pad_length = max_clue_length + 2 - len(clue)
    for j in range(pad_length):
        clue = clue + ['<PAD>']
    clues[i] = clue
    clue_list = []
    for word in clue:
        clue_list.append(word_to_index_dict[word])
    training_clue_indices.append(clue_list)

# Make the embedding matrix
embedding_matrix = np.zeros((len(word_glove_pairs_dict), glove_length))
for word in word_to_index_dict.keys():
    embedding_matrix[word_to_index_dict[word]] = np.array(word_glove_pairs_dict[word])

# Load the model
trained_model = keras.models.load_model('trained_model.h5')
encoder_layer_weights = trained_model.layers[6].get_weights()
decoder_layer_weights = trained_model.layers[7].get_weights()
dense_layer_weights = trained_model.layers[8].get_weights()

# Define the model
a_LSTM = 128

a0 = keras.layers.Input(shape = (a_LSTM,), name = 'a0')
c0 = keras.layers.Input(shape = (a_LSTM,), name = 'c0')

word_index = keras.layers.Input(shape = (1,), dtype = 'int32', name = 'word_index')
x_word = keras.layers.Embedding(len(word_glove_pairs_dict), glove_length, weights = [embedding_matrix], trainable = False, name = 'x_word')(word_index)
encoder_layer = keras.layers.LSTM(a_LSTM, return_state = True, name = 'encoder_LSTM', weights = encoder_layer_weights)
encoder_output, a, c = encoder_layer(x_word, initial_state = [a0, c0])

clue_indices = keras.layers.Input(shape = (1,), dtype = 'int32', name = 'clue_indices')
x_clue = keras.layers.Embedding(len(word_glove_pairs_dict), glove_length, weights = [embedding_matrix], trainable = False, name = 'x_clue')(clue_indices)

decoder_layer = keras.layers.LSTM(a_LSTM, return_state = True, return_sequences = True, name = 'decoder_LSTM', weights = decoder_layer_weights)
LSTM_output, _, _ = decoder_layer(x_clue, initial_state = [a, c])
dense_layer = keras.layers.TimeDistributed(keras.layers.Dense(len(word_glove_pairs_dict), weights = dense_layer_weights))
dense_output = dense_layer(LSTM_output)
output = keras.layers.Activation('softmax')(dense_output)

model = keras.models.Model(inputs = [a0, c0, word_index, clue_indices], outputs = output)

# Compile the model
#model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
#
# Summarize the model
#print(model.summary())
#
# Define the training set
#x_train_a0 = np.zeros((NUM_TRAIN, a_LSTM))
#x_train_c0 = np.zeros((NUM_TRAIN, a_LSTM))
#x_train_word_index = np.array(indices)
#x_train_clue_indices = np.array(training_clue_indices)
#x_train = [x_train_a0, x_train_c0, x_train_word_index, x_train_clue_indices]
#
#y_train = np.zeros((NUM_TRAIN, max_clue_length + 2, len(word_glove_pairs_dict)))
#for m in range(NUM_TRAIN):
#    clue = clues[m]
#    for i in range(max_clue_length + 2):
#        y_train[m, i, word_to_index_dict[clue[i]]] = 1
#
# Visualize model
#keras.utils.plot_model(model, to_file='model.png', show_shapes = True)

# Define the inference set
NUM_INFER = 1
WORD_IDX = 10
#WORD_IDX = word_to_index_dict['feet']
x_infer_a0 = np.zeros((NUM_INFER, a_LSTM))
x_infer_c0 = np.zeros((NUM_INFER, a_LSTM))
x_infer_word_index = np.array([indices[WORD_IDX]])
#x_infer_word_index = np.array([WORD_IDX])
x_infer = [x_infer_a0, x_infer_c0, x_infer_word_index]

# Define the inference setup
encoder_model = keras.models.Model(inputs = [a0, c0, word_index], outputs = [a, c])
#keras.utils.plot_model(encoder_model, to_file='encoder_model.png', show_shapes = True)

decoder_state_input_a = keras.layers.Input(shape = (a_LSTM,))
decoder_state_input_c = keras.layers.Input(shape = (a_LSTM,))
decoder_outputs, a, c = decoder_layer(x_clue, initial_state = [decoder_state_input_a, decoder_state_input_c])
decoder_outputs = dense_layer(decoder_outputs)
decoder_model = keras.models.Model(inputs = [clue_indices] + [decoder_state_input_a, decoder_state_input_c], outputs = [decoder_outputs] + [a, c])

#keras.utils.plot_model(decoder_model, to_file='decoder_model.png', show_shapes = True)

states_inferred = encoder_model.predict(x_infer)
#states_inferred = [np.random.rand(NUM_INFER, a_LSTM), x_infer_c0]
generated_clue = ['<START>']

stop_condition = False
max_length = 20
while not stop_condition:
    #print(word_to_index_dict[generated_clue[len(generated_clue) - 1]])
    output_OH, a_infer, c_infer = decoder_model.predict([np.array([word_to_index_dict[generated_clue[len(generated_clue) - 1]]])] + states_inferred)
    generated_word = index_to_word_dict[np.argmax(output_OH)]
    generated_clue.append(generated_word)
    if generated_word == '<END>' or len(generated_clue) == max_length:
        stop_condition = True
    states_inferred = [a_infer, c_infer]

print('')
print('Word:')
print(index_to_word_dict[indices[WORD_IDX]])
#print(index_to_word_dict[WORD_IDX])
print('')
print('Generated clue:')
for word in generated_clue:
    print(word)
#    print(word_to_index_dict[word])
print('')
