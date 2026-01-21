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

def backward(v,grad = None,verbrose=False,pre_topo = False):
    topo = topological_sort(v)

    if grad is not None:
        if v.grad.shape != grad.shape:
            raise RuntimeError("Incorrect gradient size")
        v.grad = grad
    else:
        v.grad = np.ones_like(v.data)

    if pre_topo:
        print(topo)

    for node in reversed(topo):
        if node.grad_fn is None:
            continue
        if node.required_grad == False:
            continue

        gradiendts = node.grad_fn.backward(node.grad)
        for i,(parent_node,parent_grad) in enumerate(zip(node.parents,gradiendts)):
            if parent_node.grad is None:
                parent_node.grad = parent_grad
            else:
                try:
                    parent_node.grad += parent_grad
                except Exception as e:
                    print("\n--- DEBUG INFO ---")
                    print("parent_node:", parent_node)
                    print("parent_node.grad:", parent_node.grad)
                    print("parent_node.grad.shape:", getattr(parent_node.grad, "shape", None))

                    print("parent_grad:", parent_grad)
                    print("parent_grad type:", type(parent_grad))
                    print("parent_grad.shape:", getattr(parent_grad, "shape", None))
                    print("------------------\n")

                    raise e


    if verbrose:
        print(topo)