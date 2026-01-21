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
    
## intelligent by not creating a 2d array of One-hot encoding 
class CrossEntropy(Loss):

    def forward(self,y_pred,y_true):
        if not isinstance(y_pred,tensor):
            y_pred = tensor(y_pred,required_grad=True)

        self.y_pred = y_pred

        self.batch_size,self.num_classes = self.y_pred.data.shape
        labels = y_true.data if isinstance(y_true, tensor) else y_true
        self.labels = labels.astype(int)
        
        expos = self.y_pred.exp()
        self.probs = expos/(expos.data.sum(axis=1, keepdims=True))
        logits = self.probs.log()
        
        self.correct_probs = logits.data[np.arange(self.batch_size),self.labels]
        loss = -self.correct_probs.mean()

        grad = self.probs.data
        grad[np.arange(self.batch_size), self.labels] -= 1
        grad /= self.batch_size

        self.error = tensor(loss,parents=[self.y_pred])
        self.y_pred.grad = grad
        return self.error

    def backward(self):
        return self.error.backward()
    
    def __call__(self,y_pred,y_true):
        return self.forward(y_pred,y_true)