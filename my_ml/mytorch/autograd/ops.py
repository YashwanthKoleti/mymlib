from .node import Op
import numpy as np

#########
# Always return gradients from backward() as a tuple:
#   (grad_a, grad_b) for binary ops
#   (grad_a,) for unary ops
#
# The autograd engine uses zip(parent_nodes, gradients),
# zip only works for iterable
# so the gradients object must be iterable.
# 
# If you return grad_a instead of (grad_a,), Python treats it as a scalar
# scalar i snot an iterable
# and zip() will raise a TypeError.
########

########
# In Forward(self,*parents),
# parents are tensor.data(basically they are numpy), 
# they are not tensors
########

def broadcast(arr,shape):
        return np.broadcast_to(arr, shape)
    
def un_broadcast(arr, shape):
        if np.isscalar(arr) or arr.shape == () :
            return np.ones_like(shape)*arr
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
        a, b = self.saved_tensors
        return (un_broadcast(grad_output,a.shape), un_broadcast(grad_output,b.shape))


class MulOp(Op):
    def forward(self, a, b):
        self.saved_tensors = (a, b)
        return a * b

    def backward(self, grad_output):
        a, b = self.saved_tensors
        grad_a = grad_output * b
        grad_b = grad_output * a
        #apply un-broadcasting and reply
        return (un_broadcast(grad_a,a.shape), un_broadcast(grad_b,b.shape))


class MatMulOp(Op):
    def forward(self, a, b):
        self.saved_tensors = (a, b)
        return a @ b

    def backward(self, grad_output):
        a, b = self.saved_tensors
        a_data = np.asarray(a.data)
        b_data = np.asarray(b.data)
        g_data = np.asarray(grad_output)

        if a_data.ndim == 1:
            a_data = a_data[None, :]

        if g_data.ndim == 1:
            g_data = g_data[None, :]

        grad_a = g_data @ b.T
        grad_b = a_data.T @ g_data
        #apply un-broadcasting and reply

        return (un_broadcast(grad_a,a.shape), un_broadcast(grad_b,b.shape))
    
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

class PowerOp(Op):
    def forward(self,a,b):
        self.saved_tensor = (a,b)
        return a**b
    
    def backward(self,grad_output):

        a,b = self.saved_tensor
        # grad_b is (a**b)*np.log(a)*grad_output
        # so we keep grad_b is 0, because sometimes a can have -ve values
        # Also, b is typically a constant exponent like 2 in MSE error**2
        return (b*(a**(b-1))*grad_output,np.zeros_like(b))
    
class MeanOp(Op):
    def forward(self,a):
        self.saved_tensor = a
        out_data = np.mean(a)
        return out_data
    
    def backward(self, grad_output):
        a = self.saved_tensor
        return un_broadcast(grad_output/np.prod(a.shape),a.shape)
    
class SumOp(Op):
    def forward(self, a):
        self.saved_tensor = a
        return np.sum(a)
    
    def backward(self, grad_output):
        a = self.saved_tensor
        return un_broadcast(grad_output,a.shape)
    
class LogOp(Op):
    def forward(self, a):
        self.saved_tensor = a
        return np.log(np.maximum(1e-15,a))
    
    def backward(self, grad_output):
        a = self.saved_tensor
        return (grad_output*(a**-1),)
    
class ExpOp(Op):
    def forward(self, a):
        self.saved_tensor = a
        return np.exp(a)
    
    def backward(self, grad_output):
        a = self.saved_tensor
        return (np.exp(a)*grad_output,)