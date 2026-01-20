import numpy as np
from ..tensor import tensor

class Module:
    
    def forward(self):
        raise NotImplementedError
    

    def __call__(self, x):
        return self.forward(x)
    
    def backward(self):
        raise NotImplementedError