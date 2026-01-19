import numpy as np

class tensor:
    def __init__(self,data,parents = None,operation = None):
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
        self.requires_grad = True
        self.parents = parents if parents is not None else []
        self.op = operation
        self._backward = lambda : 0
        
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

        new_tensor = tensor(self.data+other.data,parents=[self,other],operation='+')
        def _backward():
            self.grad += self.un_broadcast(new_tensor.grad,(self.shape))
            other.grad += self.un_broadcast(new_tensor.grad,(other.shape))

        new_tensor._backward = _backward
        return new_tensor
    
    def __mul__(self,other):
        other = other if isinstance(other,tensor) else tensor(other)

        new_tensor = tensor(self.data*other.data,parents=[self,other],operation='*')
        def _backward():
            self.grad += self.un_broadcast(other.data*new_tensor.grad,(self.shape))
            other.grad += self.un_broadcast(self.data*new_tensor.grad,(other.shape))

        new_tensor._backward = _backward
        return new_tensor 

    def __matmul__(self,other):
        other = other if isinstance(other,tensor) else tensor(other)
        data = self.data @ other.data
        new_tensor = tensor(data, parents=[self, other], operation='matmul')
        def _backward():
            self.grad += self.un_broadcast(new_tensor.grad @ other.data.T,(self.shape))
            other.grad += self.un_broadcast(self.data.T @ new_tensor.grad,(other.shape))

        new_tensor._backward = _backward
        return new_tensor
    
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

    def relu(self):
        new_tensor = tensor(np.maximum(0,self.data),parents=[self],operation = 'relu')
        def _backward():
            self.grad += new_tensor.grad * (self.data > 0)
            
        new_tensor._backward = _backward
        return new_tensor

    def sigmoid(self):
        new_tensor = tensor(1/(1+np.exp(-self.data)),parents=[self],operation='sigmoid')
        def _backward():
            sig = new_tensor.data
            self.grad += new_tensor.grad * sig * (1 - sig)

        new_tensor._backward = _backward
        return new_tensor

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
        self.grad = np.ones_like(self.data)
        topo = self.topological_sort()

        for v in reversed(topo):
            v._backward()    