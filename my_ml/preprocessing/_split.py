import numpy as np
import random

def Data_split(X,y,pert,val = False,shuffle = True):
    if val:
        if len(pert) != 2:
            raise RuntimeError('Expected 2 ratio values')
    else:
        if len(pert) != 1:
            raise RuntimeError('Expected 1 value in pert')
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        random.shuffle(indices)
        X = X[indices]
        y = y[indices]
    
    n_train = int(pert[0]*n_samples)
    if val:
        n_test = int(pert[1]*n_samples)
        n_val = n_samples - n_train - n_test
        n_train_idx,n_test_idx,n_val_idx = indices[0:n_train],indices[n_train:n_test+n_train],indices[n_train+n_test:]
        return (X[n_train_idx],y[n_train_idx]),(X[n_test_idx],y[n_test_idx]),(X[n_val_idx],y[n_val_idx])
    else:
        n_test = n_samples - n_test
        n_train_idx,n_test_idx = indices[0:n_train],indices[n_train:]
        return (X[n_train_idx],y[n_train_idx]),(X[n_test_idx],y[n_test_idx])
    