# How to : compute the distance to each word of the vocabulary
# (for which we have an embedding).
# Compute an embedding to find the nearest neighbor in the lexicon

# We first load polyglot

import pickle
import numpy as np
from levenshtein_distance import levenshtein_distance


def word_to_grammar(word, lexicon):
    return lexicon[word]


def oov_grammar(unknown_word, lexicon):

    poly_words, embeddings = pickle.load(open('data/polyglot-fr.pkl', 'rb'),
                                         encoding='latin1')

    lexicon_words = lexicon.keys()

    def embedding_distance(unknown_embedding, word):
        index = poly_words.index(word)
        embedding = embeddings[index]
        return np.linalg.norm(unknown_embedding - embedding)

    if unknown_word in lexicon_words:
        return unknown_word, word_to_grammar(unknown_word, lexicon)
    else:
        # We first try using a levenshtein distance of 1
        distances = [levenshtein_distance(unknown_word, word)
                     for word in lexicon_words]
        distances = np.array(distances)
        if (distances == 1).sum() > 0:
            return list(lexicon_words)[distances.argmin()]

        if unknown_word in poly_words:
            index = poly_words.index(unknown_word)
            unknown_embedding = embeddings[index]
            distances = [embedding_distance(unknown_embedding, word)
                         for word in lexicon_words if word in poly_words]
            common_words = [word for word in lexicon_words
                            if word in poly_words]
            distances = np.array(distances)
            best_match = np.argmin(distances)
            return common_words[best_match],\
                word_to_grammar(common_words[best_match], lexicon)
        else:
            distances = [levenshtein_distance(unknown_word, word)
                         for word in lexicon_words]
            distances = np.array(distances)
            return list(lexicon_words)[np.argmin(distances)],\
                word_to_grammar(list(lexicon_words)[np.argmin(distances)],
                                lexicon)
