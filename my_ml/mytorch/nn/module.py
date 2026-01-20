import numpy as np
from ..tensor.tensor import tensor

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = []

    def parameters(self):
        params = list(self._parameters.values())
        for submodule in self._modules:
            params.extend(submodule.parameters())
        
        return params

    def forward(self):
        raise NotImplementedError    

    def __call__(self, x):
        return self.forward(x)
    
    def __setattr__(self, name, value):
        if isinstance(value,tensor):
            self._parameters[name] = value
            super().__setattr__(name,value)
            return
        
        elif isinstance(value,Module):
            self._modules.append(value)
            super().__setattr__(name,value)
            return
        
        elif name.endswith('_'):
            super().__setattr__(name,value)
            return
        
        elif isinstance(value,list) and all(isinstance(v,Module) for v in value):
            for v in value:
                self._modules.append(v)
            super().__setattr__(name,value)
            return

        super().__setattr__(name,value)
        pass