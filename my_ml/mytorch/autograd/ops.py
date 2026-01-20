from .node import Op
import numpy as np


def broadcast(arr,shape):
        return np.broadcast_to(arr, shape)
    
def un_broadcast(arr, shape):
        if arr.shape == shape:
            return arr
    # Reduce extra dimensions
        while arr.ndim > len(shape):
            arr = arr.sum(axis=0)

    # Sum along broadcasted axes
        for i, (gdim, sdim) in enumerate(zip(arr.shape, shape)):
            if sdim == 1:
                arr = arr.sum(axis=i, keepdims=True)

        return arr.reshape(shape)

class AddOp(Op):
    def forward(self, a, b):
        self.saved_tensors = (a, b)
        return a + b

    def backward(self, grad_output):
        #apply un-broadcasting and reply
        return (un_broadcast(grad_output,self.saved_tensors[0].shape), un_broadcast(grad_output,self.saved_tensors[1].shape))


class MulOp(Op):
    def forward(self, a, b):
        self.saved_tensors = (a, b)
        return a * b

    def backward(self, grad_output):
        a, b = self.saved_tensors
        grad_a = grad_output * b
        grad_b = grad_output * a
        #apply un-broadcasting and reply
        return (un_broadcast(grad_a,self.saved_tensors[0].shape), un_broadcast(grad_b,self.saved_tensors[1].shape))


class MatMulOp(Op):
    def forward(self, a, b):
        self.saved_tensors = (a, b)
        return a @ b

    def backward(self, grad_output):
        a, b = self.saved_tensors
        grad_a = grad_output @ b.T
        grad_b = a.T @ grad_output
        #apply un-broadcasting and reply
        return (un_broadcast(grad_a,self.saved_tensors[0].shape), un_broadcast(grad_b,self.saved_tensors[1].shape))
    
class ReLUOp(Op):
    def forward(self, a):
        out = np.maximum(0, a)
        self.saved_tensors = (a,)      # store input for backward
        return out

    def backward(self, grad_output):
        (a,) = self.saved_tensors      # unpack saved input
        grad_a = grad_output * (a > 0) # elementwise derivative
        return (grad_a,)               # one parent → tuple of one element

    
class SigmoidOp(Op):
    def forward(self, a):
        sig = 1 / (1 + np.exp(-a))
        self.saved_tensors = (sig,)     # store sigmoid output
        return sig

    def backward(self, grad_output):
        (sig,) = self.saved_tensors
        grad_a = grad_output * sig * (1 - sig)
        return (grad_a,)