from oov import oov_grammar
import numpy as np
from nltk import Nonterminal
from nltk import Tree


def parser(tokens, lexicon, grammar):
    corrected_tokens = [oov_grammar(token, lexicon)[0] for token in tokens]

    print('Parsing tokens')

    non_terminal_symbols = set()
    start_symbol = Nonterminal('SENT')
    non_terminal_symbols.add(start_symbol)

    for prod in grammar.productions():
        non_terminal_symbols.add(prod.lhs())
    non_terminal_symbols = list(non_terminal_symbols)
    non_terminal_symbol_to_index = {non_terminal_symbols[i]: i
                                    for i in range(len(non_terminal_symbols))}

    n = len(corrected_tokens)
    r = len(non_terminal_symbols)

    P = np.zeros(shape=(n, n, r))
    back = [[[[] for i in range(r)] for j in range(n)] for k in range(n)]

    for s in range(1, n+1):
        for prod in grammar.productions(rhs=corrected_tokens[s-1]):
            symbol = prod.lhs()
            index = non_terminal_symbol_to_index[symbol]
            P[0, s-1, index] = prod.prob()

    for l in range(2, n+1):
        for s in range(1, n-l+2):
            for p in range(1, l):
                for prod in grammar.productions():
                    if len(prod.rhs()) > 1:
                        a = non_terminal_symbol_to_index[prod.lhs()]
                        b = non_terminal_symbol_to_index[prod.rhs()[0]]
                        c = non_terminal_symbol_to_index[prod.rhs()[1]]
                        prob_splitting = \
                            prod.prob() * P[p-1, s-1, b] * P[l-p-1, s+p-1, c]
                        if prob_splitting != 0:
                            # and P[l-1, s-1, a] < prob_splitting:
                            P[l-1, s-1, a] = prob_splitting
                            back[l-1][s-1][a] = (p, b, c)

    def get_tree(symbol, length, start, back):
        if length == 0:
            return Tree(non_terminal_symbols[symbol], [tokens[start]])
        else:
            s, b, c = back[length][start][symbol]
            return Tree(non_terminal_symbols[symbol],
                        [get_tree(b, s-1, start, back),
                         get_tree(c, length - s, start + s, back)])

    start_index = non_terminal_symbols.index(Nonterminal('SENT'))
    if P[n-1, 0, start_index] != 0:
        length = n - 1
        start = 0
        t = get_tree(non_terminal_symbol_to_index[Nonterminal('SENT')],
                     length, start, back)
        print(t.__str__().replace('\n', ''))
        return t.__str__().replace('\n', '')
    else:
        return None
