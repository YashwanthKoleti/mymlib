import numpy as np
from ..autograd.ops import AddOp,MulOp,MatMulOp,ReLUOp,SigmoidOp,PowerOp,SumOp,MeanOp,LogOp,ExpOp,MaxOp,GatherOp,TransposeOp,ReshapeOp,TanhOp,Conv1dOp,Conv2dOp,Conv3dOp,maxpool1d,maxpool2d,maxpool3d

########
# self.parents is a list,
# so when do any operation, dont forget to assign self.parents as list
######## 



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
        self.required_grad = required_grad
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
        other = other if isinstance(other,tensor) else tensor(other)
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other,tensor) else tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other,tensor) else tensor(other)
        return other + (-self)
    
    def __truediv__(self, other):
        other = other if isinstance(other,tensor) else tensor(other)
        return self * (other**-1)

    def __rtruediv__(self, other):
        other = other if isinstance(other,tensor) else tensor(other)
        return other * (self**-1)

    def __pow__(self,other):
        other = other if isinstance(other,tensor) else tensor(other)
        return PowerOp().apply(self, other)

    def relu(self):
        return ReLUOp().apply(self)

    def sigmoid(self):
        return SigmoidOp().apply(self)
    
    def tanh(self):
        return TanhOp().apply(self)

    def power(self,other):
        other = other if isinstance(other,tensor) else tensor(other)
        return PowerOp().apply(self,other)
    
    def mean(self,axis = None,keepdims = False):
        return MeanOp().apply(self,axis,keepdims=keepdims)

    def sum(self,axis = None,keepdims = False):
        return SumOp().apply(self,axis,keepdims=keepdims)
    
    def log(self):
        return LogOp().apply(self)
    
    def exp(self):
        return ExpOp().apply(self)
    
    def max(self,axis = None,keepdims = False):
        return MaxOp().apply(self,axis,keepdims=keepdims)
        
    def gather(self,labels):
        return GatherOp().apply(self,labels=labels)
    
    def transpose(self):
        return TransposeOp().apply(self)
    
    def reshape(self,new_shape):
        return ReshapeOp().apply(self,new_shape=new_shape)
    
    def Con1d(self,other,padding,stride):
        other = other if isinstance(other,tensor) else tensor(other)
        return Conv1dOp().apply(self,other,padding,stride)
    
    def Con2d(self,other,padding,stride):
        other = other if isinstance(other,tensor) else tensor(other)
        return Conv2dOp().apply(self,other,padding,stride)
    
    def Con3d(self,other,padding,stride):
        other = other if isinstance(other,tensor) else tensor(other)
        return Conv3dOp().apply(self,other,padding,stride)
    
    def Maxpool1d(self,stride,padding,kernel_size):
        return maxpool1d().apply(self, stride,padding,kernel_size)
    
    def Maxpool2d(self,stride,padding,kernel_size):
        return maxpool2d().apply(self, stride,padding,kernel_size)
    
    def Maxpool3d(self,stride,padding,kernel_size):
        return maxpool3d().apply(self, stride,padding,kernel_size)
    
    def Avgpool1d(self,padding,stride,kernel_size):
        kernel = np.ones(kernel_size, dtype=np.float32)
        kernel = kernel/kernel.size
        other = tensor(kernel)
        kernel.required_grad = False
        return Conv1dOp().apply(self,other,padding,stride)
    
    def Avgpool2d(self,padding,stride,kernel_size):
        kernel = np.ones(kernel_size, dtype=np.float32)
        kernel = kernel/kernel.size
        other = tensor(kernel)
        kernel.required_grad = False
        return Conv1dOp().apply(self,other,padding,stride)
    
    def Avgpool3d(self,padding,stride,kernel_size):
        kernel = np.ones(kernel_size, dtype=np.float32)
        kernel = kernel/kernel.size
        other = tensor(kernel)
        kernel.required_grad = False
        return Conv1dOp().apply(self,other,padding,stride)

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

    def backward(self,grad = None,verbose = False,pre_topo = False):
        if grad is not None:
            if self.grad.shape != grad.shape:
                raise RuntimeError("Incorrect gradient size")
            self.grad = grad


        from ..autograd.engine import backward
        backward(self,grad,verbose,pre_topo=pre_topo)