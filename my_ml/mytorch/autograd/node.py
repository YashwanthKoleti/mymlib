########
# Remember that we are send tensor.data in forward(self,*inputs), not tensor
########

class Op:
    def __init__(self):
        self.parents = []

    def forward(self,*inputs):
        raise NotImplementedError

    def backward(self,*grad_output):
        raise NotImplementedError
    
    def apply(self,*parents):
        from ..tensor.tensor import tensor 
        self.parents = parents

        inputs = [
                t.data if isinstance(t, tensor) else tensor(t).data
                for t in self.parents
                ]

            
        out_data = self.forward(*inputs)
        out = tensor(out_data,self.parents,grad_fn=self,required_grad=True)

        return out