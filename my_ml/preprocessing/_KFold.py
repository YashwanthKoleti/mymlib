import numpy as np
import random

# yeild is new

def KFold(X,y,shuffle = False,k_split= 5):

    if not isinstance(k_split, int) or k_split <= 1:
        raise ValueError("k_split must be an integer greater than 1.")
    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be a boolean value.")
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    remainder = n_samples%k_split
    base_size = n_samples//k_split

    fold_sizes = np.full(k_split,base_size,dtype=int)
    fold_sizes[:remainder] += 1

    current = 0
    for fold_size in fold_sizes:
        start = current
        end = start+fold_size
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        yield train_idx, val_idx
        current = end