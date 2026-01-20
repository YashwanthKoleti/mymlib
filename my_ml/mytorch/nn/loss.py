import numpy as np
from ..tensor.tensor import tensor
from .module import Module

class Loss:
    
    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError
    

class MSE(Loss):

    def forward(self,y_true,y_pred):
        if not isinstance(y_pred,tensor):
            y_pred = tensor(y_pred,required_grad=True)
        if not isinstance(y_true,tensor):
            y_true = tensor(y_true,required_grad=False)
            
        error = y_pred-y_true
        error = error.power()
        error = error/error.shape[0]

        self.error = error
        return error
    
    def backward(self):
        return self.error.backward()
    
    def __call__(self,y_true,y_pred):
        return self.forward(y_true,y_pred)