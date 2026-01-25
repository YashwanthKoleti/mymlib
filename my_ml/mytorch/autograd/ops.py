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
        arr = np.asarray(arr)
        if np.isscalar(arr) or arr.shape == ():
            return np.full(shape, arr)
        
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
    def apply(self,a,axis,keepdims = False):
        from ..tensor.tensor import tensor 
        self.parents = [a]
        self.axis = axis
        self.keepdims = keepdims
        out_data = self.forward(a.data,axis=axis,keepdims=keepdims)
        return tensor(out_data,self.parents,grad_fn=self)
    
    def forward(self,a,axis,keepdims):
        self.saved_tensor = a
        out_data = np.mean(a,axis=axis,keepdims = keepdims)
        return out_data
    
    def backward(self, grad_output):
        a = self.saved_tensor
        axis = self.axis

        # compute divisor for per-axis mean
        if axis is None:
            divisor = np.prod(a.shape)
        else:
            divisor = a.shape[axis]

        # if keepdims=False, expand dims to match original shape
        if not self.keepdims and axis is not None:
            grad_output = np.expand_dims(grad_output, axis=axis)

        # broadcast to original shape
        grad = np.ones_like(a) * (grad_output / divisor)
        return (grad,)

class SumOp(Op):
    def apply(self,a,axis,keepdims = False):
        from ..tensor.tensor import tensor 
        self.parents = [a]
        self.axis = axis
        self.keepdims = keepdims
        out_data = self.forward(a.data,axis=axis,keepdims = keepdims)
        return tensor(out_data,self.parents,grad_fn=self)
    
    def forward(self, a,axis,keepdims):
        self.saved_tensor = a
        return np.sum(a,axis = axis,keepdims=keepdims)
    
    def backward(self, grad_output):
        a = self.saved_tensor
        axis = self.axis

        # if keepdims=False, expand dims to match original
        if not self.keepdims and axis is not None:
            grad_output = np.expand_dims(grad_output, axis=axis)

        # broadcast to original shape
        grad = np.ones_like(a) * grad_output
        return (grad,)

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
    
class MaxOp(Op):
    def apply(self,a,axis,keepdims = False):
        from ..tensor.tensor import tensor 
        self.parents = [a]
        out_data = self.forward(a.data,axis=axis,keepdims=keepdims)
        return tensor(out_data,self.parents,grad_fn=self)

    def forward(self, a,axis,keepdims):
        self.saved_tensor = a
        self.axis = axis
        self.idxs = np.argmax(a,axis=axis)

        expanded = np.expand_dims(self.idxs, axis=axis)
        return np.take_along_axis(a, expanded, axis=axis)
    
    def backward(self, grad_output):
        a = self.saved_tensor
        idx = self.idxs
        axis = self.axis

        grad = np.zeros_like(a)
        expanded_idx = np.expand_dims(idx, axis=axis)

        np.put_along_axis(grad, expanded_idx, grad_output, axis=axis)

        return (grad,)

class GatherOp(Op):
    def apply(self,a,labels):
        from ..tensor.tensor import tensor
        self.parents = [a]
        out_data = self.forward(a.data,labels)
        return tensor(out_data,self.parents,grad_fn=self,required_grad=True)

    def forward(self, a,labels):
        self.saved_tensor = a
        self.labels = labels
        
        data = a[np.arange(a.shape[0]),self.labels]
        return data
    
    def backward(self, grad_output):
        labels = self.labels
        a = self.saved_tensor

        grad = np.zeros_like(a)
        grad[np.arange(len(labels)), labels] = grad_output

        return (grad,)
    
class TransposeOp(Op):
    def forward(self, a):
        return a.T
    
    def backward(self, grad_output):
        return (grad_output.T,)
    
class ReshapeOp(Op):
    def apply(self,a,new_shape):
        from ..tensor.tensor import tensor
        self.parents = [a]
        out_data = self.forward(a.data,new_shape)
        return tensor(out_data,self.parents,grad_fn=self)
    
    def forward(self, a,new_shape):
        self.saved_tensor = a
        return np.reshape(a,new_shape)
    
    def backward(self, grad_output):
        a = self.saved_tensor
        return (np.reshape(grad_output,a.shape),)
    
class TanhOp(Op):
    def forward(self, a):
        self.saved_tensor = a
        self.buffer = np.tanh(a)
        return self.buffer
    
    def backward(self, grad_output):
        a = self.saved_tensor
        return ((1-np.square(self.buffer))*grad_output,)