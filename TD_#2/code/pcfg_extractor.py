import os
import re

# Probabilitic context-free grammar extractor

from nltk import induce_pcfg
from nltk import Tree, Nonterminal

# Loading data


def get_lexicon(data_folder='../data', filename='sequoia-corpus+fct.mrg_strict'):

    path = os.path.join(data_folder, filename)

    # Make a better REGEX
    regex = re.compile(r'[-|_|\+][A-Z]+')

    trees = []
    productions = []

    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = regex.sub('', line)
            tree = Tree.fromstring(line)
            # tree.collapse_unary(collapsePOS=False)
            # tree.chomsky_normal_form(horzMarkov=2)
            trees.append(tree)
            productions += tree.productions()

    starting_symbol = Nonterminal('SENT')
    grammar = induce_pcfg(starting_symbol, productions)

    lexicon = dict()
    for prod in grammar.productions():
        if len(prod.rhs()) == 1:  # Terminal element
            word = prod.rhs()[0]
            if word in lexicon.keys():
                lexicon[word].append([prod.lhs(), prod.prob()])
            else:
                lexicon[word] = [[prod.lhs(), prod.prob()]]

    # Normalizing
    for word in lexicon.keys():
        s = 0
        for relation in lexicon[word]:
            s += relation[1]
        for relation in lexicon[word]:
            relation[1] = relation[1] / s

    return lexicon
