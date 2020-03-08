import sys
from cyk import parser
from pcfg_extractor import get_lexicon

tokens = sys.argv[1:]
lexicon, grammar = get_lexicon()

parsed = parser(tokens, lexicon, grammar)
print(parsed)
