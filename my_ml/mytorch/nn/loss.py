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
        error = error**2
        error = error.mean()

        self.error = error
        return error
    
    def backward(self):
        return self.error.backward()
    
    def __call__(self,y_true,y_pred):
        return self.forward(y_true,y_pred)
    
class CrossEntropy(Loss):

    def forward(self,y_pred,y_true):
        if not isinstance(y_pred,tensor):
            y_pred = tensor(y_pred,required_grad=True)

        self.y_pred = y_pred
        expos = self.y_pred.exp()
        demo = (expos.sum(axis=1,keepdims=True))
        probs = expos/demo
        logits = probs.log()

        error = logits.gather(y_true).mean()
        self.error = -error
        return self.error

    def backward(self):
        return self.error.backward()
    
    def __call__(self,y_pred,y_true):
        return self.forward(y_pred,y_true)