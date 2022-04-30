"""Provide utilities for the use of pyentropy or pypt.

Patricia Wollstadt
01/12/2015
"""

import numpy as np

def encode_words_jidt(x, k, alphabet_size):
    """Encode words in data strings as decimal numbers.
    
    Encode words of length k in the data string x as decimal numbers. 
    Symbols in x are assumed to come from an alphabet with the size 
    given in alphabet_size.

    This code is taken from JIDT.
    
    Args:
        x (np array): string of symbols with a max. value < 
            alphabet_size 
        k (int): word length or history
        alphabet_size (int): size of the alphabet of symbols in x
    
    Returns:
        np array, int: encoded words
        np array, int: prediction point/next point following each 
            encoded word
    
    Raises:
        ValueError: if the actual  alphabet_size of x exceeds the number 
            given in alphabet_size
    """
    if np.unique(x).shape[0] > alphabet_size:
        print("the number of unique values in x is larger than the alphabet size!")
        raise ValueError
    
    n = x.shape[0]
    observations = n - k
    max_shifted_value = np.arange(alphabet_size) * (alphabet_size ** (k - 1))
    print("returning encoded history of " + \
          str(observations) + " observations")
    past = 0
    for p in x[0:k]:
        past *= alphabet_size
        past += p
    predpoint = np.empty(observations)
    paststate = np.empty(observations)
    ind = 0
    for t in range(k, n):
        predpoint[ind] = x[t]
        paststate[ind] = past
        past -= max_shifted_value[x[t-k]]
        past *= alphabet_size
        past += x[t]
        ind += 1
    return paststate.astype(int), predpoint.astype(int)


