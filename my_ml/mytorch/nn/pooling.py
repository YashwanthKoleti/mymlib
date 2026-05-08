import numpy as np
from ..tensor.tensor import tensor
from .module import Module

class MaxPool1d(Module):
    def __init__(self,stride,padding,kernel_size):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self,x):
        if not isinstance(x, tensor):
            x = tensor(x)

        if np.isscalar(x.data):
            x.data = np.array([x.data])

        return x.Maxpool1d(padding=self.padding,stride=self.stride,kernel_size=self.kernel_size)
    
    def __repr__(self):
        return (
            f"MaxPool1d(kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding})"
        )

class MaxPool2d(Module):
    def __init__(self,stride,padding,kernel_size):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self,x):
        if not isinstance(x, tensor):
            x = tensor(x)

        if np.isscalar(x.data):
            x.data = np.array([x.data])

        return x.Maxpool2d(padding=self.padding,stride=self.stride,kernel_size=self.kernel_size)
    
    def __repr__(self):
        return (
            f"MaxPool2d(kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding})"
        )


class MaxPool3d(Module):
    def __init__(self,stride,padding,kernel_size):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self,x):
        if not isinstance(x, tensor):
            x = tensor(x)

        if np.isscalar(x.data):
            x.data = np.array([x.data])

        return x.Maxpool3d(padding=self.padding,stride=self.stride,kernel_size=self.kernel_size)
    
    def __repr__(self):
        return (
            f"MaxPool3d(kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding})"
        )
    

class AvgPool1d(Module):
    def __init__(self,stride,padding,kernel_size):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self,x):
        if not isinstance(x, tensor):
            x = tensor(x)

        if np.isscalar(x.data):
            x.data = np.array([x.data])

        return x.Avgpool1d(padding=self.padding,stride=self.stride,kernel_size=self.kernel_size)
    
    def __repr__(self):
        return (
            f"AvgPool1d(kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding})"
        )

class AvgPool2d(Module):
    def __init__(self,stride,padding,kernel_size):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self,x):
        if not isinstance(x, tensor):
            x = tensor(x)

        if np.isscalar(x.data):
            x.data = np.array([x.data])

        return x.Avgpool2d(padding=self.padding,stride=self.stride,kernel_size=self.kernel_size)
    
    def __repr__(self):
        return (
            f"AvgPool2d(kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding})"
        )


class AvgPool3d(Module):
    def __init__(self,stride,padding,kernel_size):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self,x):
        if not isinstance(x, tensor):
            x = tensor(x)

        if np.isscalar(x.data):
            x.data = np.array([x.data])

        return x.Avgpool3d(padding=self.padding,stride=self.stride,kernel_size=self.kernel_size)
    
    def __repr__(self):
        return (
            f"AvgPool3d(kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding})"
        )