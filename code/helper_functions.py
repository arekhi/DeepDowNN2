def choose_word_clue_pairs(NUM_TRAIN, word_clue_pairs_list, word_glove_pairs_dict, word_to_index_dict):

    # Make a new list: for the word-clue pairs whose words appear in the word-glove 
    #   dict, translate that pair into a pair [emb_word, emb_clue_list], where
    #   emb_clue_list is the list [emb_clue_word_0, emb_clue_word_1, ...].
#    word_clue_embeddings_list = []
    num_pairs_added = 0
    done_flag = 0
    words = []
    indices = []
    clues = []
    words_present = 0
    for pair in word_clue_pairs_list:
        if done_flag:
            break
        word = pair[1].lower()
        if word in word_glove_pairs_dict:
            words_present += 1
            clue = pair[0].lower()
            for ch in [',', ':', '.', ';', '"', '\'', '!', '?', '$', '%', '(', ')']:
                clue = clue.replace(ch, '')
            clue = clue.replace('-', ' ')
            clue = clue.split()
            missing_word_flag = 0
            for clue_word in clue:
                if clue_word not in word_glove_pairs_dict:
                    missing_word_flag = 1
                    break
            if (not missing_word_flag) and (len(clue) != 0):
                words.append(word)
                indices.append(word_to_index_dict[word])
                word_embedding = word_glove_pairs_dict[word]
                clues.append(clue)
                num_pairs_added += 1
                if num_pairs_added >= NUM_TRAIN:
                    done_flag = 1
#                if (num_pairs_added % 100000) == 0:
#                    print(num_pairs_added)
#    print(words_present) 
    max_clue_length = 0
    for clue in clues:
        if len(clue) > max_clue_length:
            max_clue_length = len(clue)

    return words, indices, clues, num_pairs_added, max_clue_length

def choose_word_clue_pairs_with_dict(NUM_TRAIN, word_clue_pairs_list, word_glove_pairs_dict, word_to_index_dict, word_defn_pairs_dict):

    # Make a new list: for the word-clue pairs whose words appear in the word-glove 
    #   dict, translate that pair into a pair [emb_word, emb_clue_list], where
    #   emb_clue_list is the list [emb_clue_word_0, emb_clue_word_1, ...].
    import re
    num_pairs_added = 0
    done_flag = 0
    words = []
    indices = []
    clues = []
    definitions = []
    words_present = 0
    max_clue_length = 0
    max_defn_length = 0
    for pair in word_clue_pairs_list:
        if done_flag:
            break
        word = pair[1].lower()
        if (word in word_glove_pairs_dict) and (word in word_defn_pairs_dict):
            words_present += 1
            clue = pair[0].lower()
            defn = word_defn_pairs_dict[word].lower()
            for ch in [',', ':', '.', ';', '"', '\'', '!', '?', '$', '%', '(', ')']:
                clue = clue.replace(ch, '')
                defn = defn.replace(ch, '')
            clue = clue.replace('-', ' ')
            defn = defn.replace('-', ' ')
            defn = re.sub('<(.*?)>', '', defn)
#            print(clue)
            clue = clue.split()
            defn = defn.split()
            missing_word_flag = 0
            for clue_word in clue:
                if clue_word not in word_glove_pairs_dict:
                    missing_word_flag = 1
                    break
            if not missing_word_flag:
                for defn_word in defn:
                    if defn_word not in word_glove_pairs_dict:
                        missing_word_flag = 1
                        break
            if not missing_word_flag:
                words.append(word)
                indices.append(word_to_index_dict[word])
                word_embedding = word_glove_pairs_dict[word]
                clues.append(clue)
                if len(clue) > max_clue_length:
                    max_clue_length = len(clue)
                definitions.append(defn)
                if len(defn) > max_defn_length:
                    max_defn_length = len(defn)
                num_pairs_added += 1
                if num_pairs_added >= NUM_TRAIN:
                    done_flag = 1
    return words, indices, clues, definitions, num_pairs_added, max_clue_length, max_defn_length

def add_tokens(word_glove_pairs_dict, word_to_index_dict, index_to_word_dict, glove_length, clues, max_clue_length, np):

    # Add start, end, and pad tokens to each dictionary
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
    
    # Add start, end, and (as necessary) pad tokens to each clue
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
 
    return word_glove_pairs_dict, word_to_index_dict, index_to_word_dict, training_clue_indices, clues

def add_tokens_with_dict(word_glove_pairs_dict, word_to_index_dict, index_to_word_dict, glove_length, clues, max_clue_length, np, definitions, MAX_DEFN_LEN):

    # Add start, end, and pad tokens to each dictionary
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
    
    # Add start, end, and (as necessary) pad tokens to each clue
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
 
    # Clip and pad definitions
    definition_indices = []
    for i in range(len(definitions)):
        defn = definitions[i]
        if len(defn) > MAX_DEFN_LEN:
            defn = defn[0:MAX_DEFN_LEN]
        pad_length = MAX_DEFN_LEN - len(defn)
        for j in range(pad_length):
            defn = defn + ['<PAD>']
        definitions[i] = defn
        defn_list = []
        for word in defn:
            defn_list.append(word_to_index_dict[word])
        definition_indices.append(defn_list)

    return word_glove_pairs_dict, word_to_index_dict, index_to_word_dict, training_clue_indices, definition_indices, clues, definitions
