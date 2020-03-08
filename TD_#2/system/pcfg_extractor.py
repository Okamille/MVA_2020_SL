import os
import re

# Probabilitic context-free grammar extractor

from nltk import induce_pcfg
from nltk import Tree, Nonterminal

# Loading data


def get_lexicon(data_folder='data',
                filename='sequoia-corpus+fct.mrg_strict'):

    path = os.path.join(data_folder, filename)

    # Make a better REGEX
    regex = re.compile(r'[-|_|\+][A-Z]{2,5}(?!\))')

    trees = []
    productions = []

    with open(path, 'r') as f:
        lines = f.readlines()
        n_lines = len(lines)
        train_index = int(n_lines * 0.8)
        # We only want to train on 80% of the dataset
        for line in lines[:train_index]:
            line = regex.sub('', line)
            tree = Tree.fromstring(line)
            tree.chomsky_normal_form()
            trees.append(tree)
            productions += tree.productions()

    starting_symbol = Nonterminal('')
    grammar = induce_pcfg(starting_symbol, productions)

    lexicon = dict()
    for prod in grammar.productions():
        if len(prod.rhs()) == 1:  # Terminal element
            word = str(prod.rhs()[0]).lower()
            try:
                if word in lexicon.keys():
                    new_symbol = prod.lhs()
                    already_in = False
                    for i, (symbol, prob) in enumerate(lexicon[word]):
                        if symbol == new_symbol:
                            lexicon[word][i][1] += prod.prob()
                            already_in = True
                    if not already_in:
                        lexicon[word].append([new_symbol, prod.prob()])
                else:
                    lexicon[word] = [[prod.lhs(), prod.prob()]]
            except AttributeError:
                continue

    # Normalizing
    for word in lexicon.keys():
        s = 0
        for relation in lexicon[word]:
            s += relation[1]
        for relation in lexicon[word]:
            relation[1] = relation[1] / s

    return lexicon, grammar
