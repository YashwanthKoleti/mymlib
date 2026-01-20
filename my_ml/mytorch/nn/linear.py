import numpy as np
from ..tensor.tensor import tensor
from .module import Module

class Linear(Module):
    def __init__(self,input_dim,output_dim,bias = True):
        self.weights = tensor(np.ones((input_dim,output_dim)))
        if bias:
            self.bias = tensor(np.ones(output_dim))
        else:
            self.bias = tensor(np.zeros(output_dim),required_grad = False)
    
    def forward(self,x):
        if not isinstance(x, tensor):
            x = tensor(x)

        if np.isscalar(x.data):
            x.data = np.array([x.data])

        return x@self.weights + self.bias
    
    def __repr__(self):
        return f"Linear(in_features={self.weights.data.shape[0]}, " \
           f"out_features={self.weights.data.shape[1]}, " \
           f"bias={self.bias is not None})"
