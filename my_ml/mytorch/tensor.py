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
        self.parents = parents
        self.op = operation
        self._backward = lambda : 0
        
    @staticmethod
    def broadcast(arr,shape):
        return np.broadcast_to(arr, shape)
    @staticmethod
    def un_broadcast(arr, original_shape):
        extra_dims = arr.ndim - len(original_shape)
        for _ in range(extra_dims):
            arr = arr.sum(axis=0, keepdims=False)

        for axis, size in enumerate(original_shape):
            if size == 1:
                arr = arr.sum(axis=axis, keepdims=True)

        return arr.reshape(original_shape)

    def __add__(self,other):
        new_tensor = tensor(self.data+other.data,parents=[self,other],operation='+')
        def _backward():
            self.grad += self.un_broadcast(new_tensor.grad,(self.shape))
            other.grad += self.un_broadcast(new_tensor.grad,(other.shape))

        new_tensor._backward = _backward
        return new_tensor
    
    def __mul__(self,other):
        new_tensor = tensor(self.data*other.data,parents=[self,other],operation='*')
        def _backward():
            self.grad += self.un_broadcast(other.data*new_tensor.grad,(self.shape))
            other.grad += self.un_broadcast(self.data*new_tensor.grad,(other.shape))

        new_tensor._backward = _backward
        return new_tensor 

    def __matmul__(self,other):
        data = self.data @ other.data
        new_tensor = tensor(data, parents=[self, other], operation='matmul')
        def _backward():
            self.grad += self.un_broadcast(new_tensor.grad @ other.data.T,(self.shape))
            other.grad += self.un_broadcast(self.data.T @ new_tensor.grad,(other.shape))

        new_tensor._backward = _backward
        return new_tensor
    
    def relu(self):
        new_tensor = tensor(np.maximum(0,self.data),parents=[self],operation = 'relu')
        def _backward():
            self.grad += new_tensor.grad * (new_tensor.data > 0)
            
        new_tensor._backward = _backward
        return new_tensor

    def sigmoid(self):
        new_tensor = tensor(1/(1+np.exp(-self.data)),parents=[self],operation='sigmoid')
        def _backward():
            dummy = np.exp(-self.data)
            self.grad += new_tensor.grad * ((dummy)/((1+dummy)**2))
        new_tensor._backward = _backward
        return new_tensor

    def topological_sort(self,visited = None):
        if visited == None:
            visited = set()
        while self not in visited:
            visited.add(self)
            for parent in self.parents:
                self.topological_sort(parent,visited=visited)
        
        return visited

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



    def backward(self):
        nodes = []
        nodes.append(self)

        topo = self.topological_sort(self)
        for v in reversed(topo):
            v._backward()