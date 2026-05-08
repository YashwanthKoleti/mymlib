import numpy as np
from ..tensor.tensor import tensor
from .module import Module

class MaxPool1d(Module):
    def __init__():
        pass

class AvgPool1d(Module):
    def __init__(self,kernel_size,padding,stride,return_indices=False):
        
        pass
    
    def forward(self,x):
        if not isinstance(x, tensor):
            x = tensor(x)

        if np.isscalar(x.data):
            x.data = np.array([x.data])

        return x.Con1d(self.kernel,self.padding,self.stride)