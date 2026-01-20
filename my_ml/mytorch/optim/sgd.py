import numpy as np
from ..tensor.tensor import tensor
from .optimizer import Optimizer
from ..nn.module import Module

class SGD:
    def __init__(self,module,learning_rate = 1e-4):
        self.lr = learning_rate
        self.module = module
    
    