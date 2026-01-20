import numpy as np
from ..autograd.ops import AddOp,MulOp,MatMulOp,ReLUOp,SigmoidOp,PowerOp,SumOp,MeanOp

class tensor:
    def __init__(self,data,parents = None,grad_fn = None,required_grad = True):
        if not isinstance(data, np.ndarray):
            try:
                self.data = np.array(data, dtype=np.float64)
            except TypeError:
                raise TypeError(
                    f"Data must be convertible to a numpy array, got {type(data)}"
                )
        else:
            self.data = data

        self.shape = self.data.shape
        self.grad = np.zeros(self.shape)
        self.requires_grad = required_grad
        self.parents = parents if parents is not None else []
        self.grad_fn = grad_fn
        
    @staticmethod
    def broadcast(arr,shape):
        return np.broadcast_to(arr, shape)
    
    @staticmethod
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


    def __add__(self,other):
        other = other if isinstance(other,tensor) else tensor(other)
        return AddOp().apply(self,other)
        
    def __mul__(self,other):
        other = other if isinstance(other,tensor) else tensor(other)
        return MulOp().apply(self,other)

    def __matmul__(self,other):
        other = other if isinstance(other,tensor) else tensor(other)
        return MatMulOp().apply(self,other)
    
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)
    
    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def relu(self):
        return ReLUOp().apply(self)

    def sigmoid(self):
        return SigmoidOp().apply(self)

    def power(self):
        return PowerOp().apply(self)
    
    def mean(self):
        return MeanOp().apply(self)

    def sum(self):
        return SumOp().apply(self)

    def __repr__(self):
        def indent(value, spaces=4):
            s = str(value)
            pad = " " * spaces
            return pad + s.replace("\n", "\n" + pad)

        return (
        "tensor(\n"
        f"  data=\n{indent(self.data, 4)},\n"
        f"  grad=\n{indent(self.grad, 4)}\n"
        ")"
        )

    def topological_sort(self):
        visited = set()
        order = []

        def build(v):
            if v is None:
                return
            if v not in visited:
                visited.add(v)
                for parent in v.parents:
                    build(parent)
                order.append(v)

        build(self)
        return order

    def backward(self):
        from ..autograd.engine import backward
        backward(self)