import numpy as np


def levenshtein_distance(chaine1, chaine2):
    n = len(chaine1)
    m = len(chaine2)

    d = np.zeros(shape=(n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        d[i, 0] = i
    for j in range(m + 1):
        d[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if chaine1[i-1] == chaine2[j-1]:
                cout = 0
            else:
                cout = 1

            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + cout)
    return d[n, m]
