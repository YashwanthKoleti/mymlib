import numpy as np
from ..tensor.tensor import tensor
from .module import Module

class Flatten(Module):
    def __init__(self,input_dim,output_dim,bias = True):
        super().__init__()
    
    def forward(self,x,start_dim = 0,end_dim = -1):
        if not isinstance(x, tensor):
            x = tensor(x)

        if np.isscalar(x.data):
            x.data = np.array([x.data])

        shape = list(x.shape)
        if end_dim < 0:
            end_dim += len(shape)

        flat_size = 1
        for i in range(start_dim, end_dim + 1):
            flat_size *= shape[i]
        
        #python using + on lists performs concatenation
        new_shape = (
            shape[:start_dim] + [flat_size] + shape[end_dim + 1:]
        )

        return x.reshape(tuple(new_shape))

    def __repr__(self):
        return f"Reshape(shape={self.shape})"