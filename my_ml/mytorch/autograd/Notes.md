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


-----------------------
-- NOTE ABOUT MAXPOOL
-----------------------

We implement max pooling using:

```python
numpy.lib.stride_tricks.sliding_window_view
```

instead of explicit Python loops.

The idea is:

1. Create a **view** of all pooling windows
2. Apply `max(axis=-1)` (or over spatial axes)
3. Store `argmax` indices for backpropagation

Example:

```python
x = [1,5,2,3]
kernel = 2
stride = 1
```

Windows become:

```text
[1,5]
[5,2]
[2,3]
```

and maxpool output is:

```text
[5,5,3]
```

without manually iterating over slices.

-------------------------------------------------
-- HOW `sliding_window_view` ACTUALLY WORKS
-------------------------------------------------

`sliding_window_view` does NOT copy memory.

It creates a new tensor VIEW using NumPy strides.

Example:

```python
from numpy.lib.stride_tricks import sliding_window_view

x = np.array([1,5,2,3])

w = sliding_window_view(x, window_shape=2)
```

Result:

```text
[
 [1,5],
 [5,2],
 [2,3]
]
```

Internally this is just a reinterpretation of memory.

No windows are physically allocated.

This makes pooling significantly faster and more memory efficient.

-------------------------------------------------
-- STRIDE HANDLING
-------------------------------------------------

`sliding_window_view` itself generates ALL windows.

Stride is applied afterwards:

```python
windows = windows[::stride]
```

Example:

```python
stride = 2
```

keeps:

```text
window 0
window 2
window 4
...
```

This reproduces pooling stride behavior.

-------------------------------------------------
-- WHY WE STORE `argmax`
-------------------------------------------------

Forward pass computes:

```text
y_i = max(window_i)
```

During backward pass we must know WHICH element won.

Therefore we store:

```python
self.argmax = np.argmax(windows, axis=-1)
```

Example:

```text
window = [1,5]
argmax = 1
```

meaning index `1` inside the window received the output.

-------------------------------------------------
-- BACKPROPAGATION
-------------------------------------------------

For each pooled output:

```text
grad_output[i]
```

the gradient flows ONLY to the max element.

Index reconstruction:

```python
idx = i * stride + argmax[i]
```

Then:

```python
a_grad[idx] += grad_output[i]
```

-------------------------------------------------
-- OVERLAPPING WINDOWS
-------------------------------------------------

If:

```text
stride < kernel_size
```

windows overlap.

The SAME input element may become the max in multiple windows.

Example:

```text
[1,5]
[5,2]
```

The value `5` wins twice.

Therefore gradients ACCUMULATE:

```python
a_grad[idx] += grad_output[i]
```

instead of assignment.

This is required by the chain rule.

-------------------------------------------------
-- WHY `np.add.at` IS USEFUL
-------------------------------------------------

Repeated indices are problematic in NumPy fancy indexing.

This is WRONG:

```python
a_grad[indices] += grad_output
```

because repeated indices may overwrite updates.

Correct accumulation uses:

```python
np.add.at(a_grad, indices, grad_output)
```

which performs proper scatter-add semantics.

-------------------------------------------------
-- IMPORTANT PERFORMANCE NOTE
-------------------------------------------------

Pooling is highly parallelizable because:

* each window is independent
* reductions (`max`, `mean`) are vectorized
* NumPy executes reductions in optimized C

Therefore:

```python
windows.max(axis=-1)
```

is MUCH faster than Python loops.

Modern frameworks further optimize pooling using:

* SIMD instructions
* CUDA kernels
* shared GPU memory
* fused reductions

-------------------------------------------------
-- WHY THERE IS NO `MaxPoolBackwardOp`
-------------------------------------------------

Pooling backward is implemented directly inside:

```python
maxpool.backward()
```

because:

* gradients depend on stored argmax locations
* backward is a scatter-add operation
* no learnable parameters exist

Therefore a separate backward graph node is unnecessary.