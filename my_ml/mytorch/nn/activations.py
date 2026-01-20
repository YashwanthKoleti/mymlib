from ..tensor import tensor
import numpy as np
from .module import Module
from ..autograd.ops import ReLUOp,SigmoidOp

class ReLU(Module):
    def forward(self, x):
        self.saved_tensor = ReLUOp().apply(x)
        return self.saved_tensor

    def __repr__(self):
        return "ReLU()"


class Sigmoid(Module):
    def forward(self, x):
        self.saved_tensor = SigmoidOp().apply(x)
        return self.saved_tensor
        
    def __repr__(self):
        return "Sigmoid()"