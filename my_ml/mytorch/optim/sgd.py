import numpy as np
from ..tensor.tensor import tensor
from .optimizer import Optimizer
from ..nn.module import Module

class SGD:
    def __init__(self,param,lr = 1e-4):
        self.lr = lr
        self.parameters = param
    
    def zero_grad(self):
        for para in self.parameters:
            para.grad[:] = 0.0

    def step(self):
        for para in self.parameters:
            para.data -= self.lr*para.grad