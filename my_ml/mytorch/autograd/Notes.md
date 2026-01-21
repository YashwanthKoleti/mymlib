-- NOTE ABOUT SUBTRACTION
-----------------------

We do **NOT** implement a dedicated `SubOp`.

Subtraction is handled through addition and negation:

```text
a - b   →   a + (-b)
```

Negation (`__neg__`) is implemented as:

```text
(-b)   →   (-1) * b
```

This automatically creates a new tensor whose parents are:

```text
parents = [tensor(-1), b]
```

Because the negation is just a multiplication by `-1`, the gradient flows naturally to both parents via `MulOp`:

* gradient w.r.t. `b` is multiplied by `-1`
* gradient w.r.t. the constant `(-1)` is normally ignored

Therefore:

* **No special `SubOp` is required**
* **The autograd graph is correct by construction**
* **All gradients backpropagate properly**
* **Simillarly you dont need `DivOp`, `NegOp`**


