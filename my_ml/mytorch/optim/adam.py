import numpy as np
from ..tensor.tensor import tensor
from .optimizer import Optimizer
from ..nn.module import Module

class Adam:
    def __init__(self,param,lr = 1e-4,betas = (0.9,0.99),eps = 1e-8):
        self.lr = lr
        self.parameters = param
        self.b1,self.b2 = betas
        self.eps = eps
        self.f_momentum = []
        for t in self.parameters:
            self.f_momentum.append(np.zeros_like(t.grad))
        
        self.s_momentum = self.f_momentum



    def zero_grad(self):
        for para in self.parameters:
            para.grad[:] = 0.0

    def step(self):

        for para, m1, m2 in zip(self.parameters, self.f_momentum, self.s_momentum):
            m1 = self.b1*(m1) + (1-self.b1)*para.grad
            m2 = self.b2*m2 + (1-self.b2)*(np.square(para.grad))

            m1_n = m1/(1-self.b1)
            m2_n = m2/(1-self.b2)
            
            para.data -= self.lr*(m1_n/(self.eps + np.sqrt(m2_n)))