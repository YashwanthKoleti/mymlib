from .node import Op
import numpy as np
from ..tensor.tensor import tensor
from .ops import AddOp,MulOp,MatMulOp,ReLUOp,SigmoidOp

def topological_sort(v):
    visited = set()
    node = []

    def built(v):
        if v is None:
                return
        while v not in visited:
            visited.add(v)
            for parent in v.parents:
                built(parent)
            node.append(v)
    
    built(v)
    return node

def backward(v):
    topo = topological_sort(v)
    v.grad = np.ones_like(v.data)

    for node in reversed(topo):
        if node.grad_fn is None:
            continue

        gradiendts = node.grad_fn.backward(node.grad)
        for i,(parent_node,parent_grad) in enumerate(zip(node.parents,gradiendts)):
            parent_node.grad += parent_grad