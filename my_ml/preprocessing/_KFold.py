import numpy as np
import random

# yeild is new to me, read about it

class KFold:
    def _init_(self,k_split = 5,shuffle = False):
        if not isinstance(self.k_split, int) or self.k_split <= 1:
            raise ValueError("k_split must be an integer greater than 1.")
        if not isinstance(self.shuffle, bool):
            raise TypeError("shuffle must be a boolean value.")
        self.k_split = k_split
        self.shuffle = shuffle

    def split(self,X):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
    
        remainder = n_samples%self.k_split
        base_size = n_samples//self.k_split

        fold_sizes = np.full(self.k_split,base_size,dtype=int)
        fold_sizes[:remainder] += 1

        current = 0
        for fold_size in fold_sizes:
            start = current
            end = start+fold_size
            val_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])

            yield train_idx, val_idx
            current = end