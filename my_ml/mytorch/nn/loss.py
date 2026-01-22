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

    def forward(self, y_pred, y_true):
        # y_pred: [batch, num_classes] raw logits
        # y_true: [batch] class indices

        if not isinstance(y_pred, tensor):
            y_pred = tensor(y_pred, required_grad=True)

        self.y_pred = y_pred
        self.y_true = y_true

        # log-sum-exp trick (stable softmax)
        max_vals = y_pred.max(axis=1, keepdims=True)
        shifted = y_pred - max_vals
        log_sum_exp = (shifted.exp().sum(axis=1, keepdims=True)).log()
        log_probs = shifted - log_sum_exp  # log softmax

        correct_log_probs = log_probs.gather(y_true)

        loss = -correct_log_probs.mean()
        self.error = loss
        return loss

    def backward(self):
        return self.error.backward()
    
    def __call__(self,y_pred,y_true):
        return self.forward(y_pred,y_true)
