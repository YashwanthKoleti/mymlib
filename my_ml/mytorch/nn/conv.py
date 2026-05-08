import numpy as np
from ..tensor.tensor import tensor
from .module import Module

class Con1d(Module):
    def __init__(self,in_channel,out_channel,kernel_size,padding,stride,bias = True):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        if bias:
            self.bias = tensor(np.zeros((out_channel,1)))
        else:
            self.bias = None

        self.kernel = tensor(np.random.randn(
            out_channel,
            in_channel,
            kernel_size,
            kernel_size
        ) * np.sqrt(1.0 / (in_channel * kernel_size * kernel_size)))
        
    def forward(self,x):
        if not isinstance(x, tensor):
            x = tensor(x)

        if np.isscalar(x.data):
            x.data = np.array([x.data])

        return x.Con1d(self.kernel,self.padding,self.stride) + self.bias
    
    def __repr__(self):
        return (
            f"Con1d(in_channel={self.in_channel}, "
            f"out_channel={self.out_channel}, "
            f"kernel_size={self.kernel_size}, "
            f"padding={self.padding}, "
            f"stride={self.stride}, "
            f"bias={self.bias is not None})"
        )

class Con2d(Module):
    def __init__(self,in_channel,out_channel,kernel_size,padding,stride,bias = True):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        if bias:
            self.bias = tensor(np.zeros((out_channel,1,1)))
        else:
            self.bias = None

        self.kernel = tensor(np.random.randn(
            out_channel,
            in_channel,
            kernel_size,
            kernel_size
        ) * np.sqrt(1.0 / (in_channel * kernel_size * kernel_size)))
        
    def forward(self,x):
        if not isinstance(x, tensor):
            x = tensor(x)

        if np.isscalar(x.data):
            x.data = np.array([x.data])

        return x.Con2d(self.kernel,self.padding,self.stride) + self.bias

    def __repr__(self):
        return (
            f"Con2d(in_channel={self.in_channel}, "
            f"out_channel={self.out_channel}, "
            f"kernel_size={self.kernel_size}, "
            f"padding={self.padding}, "
            f"stride={self.stride}, "
            f"bias={self.bias is not None})"
        )

class Con3d(Module):
    def __init__(self,in_channel,out_channel,kernel_size,padding,stride,bias = True):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        if bias:
            self.bias = tensor(np.zeros((out_channel,1,1,1)))
        else:
            self.bias = None

        self.kernel = tensor(np.random.randn(
            out_channel,
            in_channel,
            kernel_size,
            kernel_size
        ) * np.sqrt(1.0 / (in_channel * kernel_size * kernel_size)))
        
    def forward(self,x):
        if not isinstance(x, tensor):
            x = tensor(x)

        if np.isscalar(x.data):
            x.data = np.array([x.data])

        return x.Con3d(self.kernel,self.padding,self.stride) + self.bias
    
    def __repr__(self):
        return (
            f"Con3d(in_channel={self.in_channel}, "
            f"out_channel={self.out_channel}, "
            f"kernel_size={self.kernel_size}, "
            f"padding={self.padding}, "
            f"stride={self.stride}, "
            f"bias={self.bias is not None})"
        )