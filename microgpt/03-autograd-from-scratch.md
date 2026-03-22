# Chapter 3: Autograd from Scratch

## What you'll learn

- What a computation graph is and how `Value` builds one node by node during the forward pass
- What "local gradients" are and why each operation stores them eagerly at construction time
- How the chain rule is mechanically applied by walking the graph in reverse (the backward pass)
- Why topological sort is needed before traversing the graph backward
- Why gradient accumulation uses `+=` instead of `=`
- What `__slots__` buys you and why it matters at scale

---

## Prerequisites

- Completed Chapter 1 (setup) and Chapter 2 (tokenization)
- Comfortable with Python classes, dunder methods, and recursion
- Basic calculus: you know that the derivative of `x^2` is `2x` — that is enough

---

## Background: Why does autograd exist?

Training a neural network means adjusting thousands (or billions) of parameters so that a loss function decreases. To know which direction to adjust each parameter, you need the partial derivative of the loss with respect to that parameter — i.e., the gradient.

Computing those derivatives by hand is impractical. Automatic differentiation (autograd) does it for you by recording every arithmetic operation during the forward pass and replaying those records in reverse during the backward pass, applying the chain rule at each step.

This technique — working backward from the output — is called **reverse-mode automatic differentiation**, or simply **backpropagation** in the neural network literature. Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) is the direct inspiration for the implementation you are about to study.

---

## The computation graph

Every time you perform an arithmetic operation on a `Value`, you create a new `Value` node that records:

1. The **result** of the operation (`data`)
2. Which nodes were its **inputs** (`_children`)
3. The **local gradient** of the output with respect to each input (`_local_grads`)

Together, these links form a directed acyclic graph (DAG) called the **computation graph**. Leaf nodes are your inputs (e.g., model parameters). The root node is your final scalar output (e.g., the loss).

Here is a tiny example: `c = a * b + a` where `a = 2`, `b = 3`.

```
Forward pass builds this graph:

  a=2 ──┬──────────────────────┐
        │                      │
        │  b=3                 │
        │   │                  │
        └── * ──► (a*b)=6      │
              local_grads=(3,2) │
                    │           │
                    └─── + ────┘──► c=8
                         local_grads=(1,1)
```

Every arrow points from child to parent. During the backward pass you will walk these arrows in reverse, from `c` back to `a` and `b`.

---

## The `Value` class — line by line

Source: [`microgpt.py` lines 30–72](../microgpt.py)

### Slots and initialization (lines 30–37)

```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads
```

**`__slots__`** tells Python not to create a per-instance `__dict__`. Normally every Python object carries a dictionary so you can attach arbitrary attributes at runtime. That dictionary costs memory — roughly 200–400 bytes per object. When you have millions of `Value` nodes in a computation graph (which you do when training a neural network), this saving is significant.

The tradeoff: you can only use the four declared attributes. Trying to set `v.foo = 1` on a slotted `Value` raises `AttributeError`.

**`grad`** starts at `0`. It will accumulate the partial derivative of the final output with respect to this node once `backward()` is called.

**`_local_grads`** is a tuple of scalars (plain Python `float` or `int`, not `Value` objects). This is important: local gradients are computed eagerly, at node creation time, using the raw `.data` values of the inputs. They do not form their own sub-graph.

### Arithmetic operations and their local gradients

Each operation constructs a new `Value` whose `_local_grads` encode the partial derivative of that operation's output with respect to each of its inputs.

#### Addition (line 39–41)

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data + other.data, (self, other), (1, 1))
```

If `out = a + b`, then:

- `d(out)/d(a) = 1`
- `d(out)/d(b) = 1`

Addition is a "gradient highway": it passes the upstream gradient through unchanged to both inputs. The local gradients are always `(1, 1)` regardless of the actual values of `a` and `b`.

The `isinstance` check lets you write `v + 3` without having to wrap `3` in a `Value` yourself. The reflected method `__radd__` handles `3 + v`.

#### Multiplication (lines 43–45)

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data * other.data, (self, other), (other.data, self.data))
```

If `out = a * b`, then:

- `d(out)/d(a) = b`
- `d(out)/d(b) = a`

The local gradients are the swapped operand values. This is why `_local_grads` stores plain numbers: we capture `other.data` and `self.data` at construction time, not references to the `Value` nodes.

```
Example: a=2, b=3, out=a*b=6
  _children    = (a, b)
  _local_grads = (3, 2)   ← b.data and a.data, captured at call time
```

#### Power (line 47)

```python
def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
```

If `out = x^n`, then `d(out)/d(x) = n * x^(n-1)` — the standard power rule. Note that `other` (the exponent) is a plain Python number here, not a `Value`. The class does not differentiate through the exponent, only through the base.

#### Natural log (line 48)

```python
def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
```

`d(ln x)/d(x) = 1/x`. Used to compute cross-entropy loss in the training loop: `loss_t = -probs[target_id].log()`.

#### Exponential (line 49)

```python
def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
```

`d(e^x)/d(x) = e^x`. The derivative of the exponential is itself — one of the few functions with this property. The value `math.exp(self.data)` is computed twice (once for `data`, once for `_local_grads`) but the result is identical.

#### ReLU (line 50)

```python
def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
```

ReLU (Rectified Linear Unit) clamps negative values to zero. Its derivative is a step function:

- `d(relu(x))/d(x) = 1` if `x > 0`
- `d(relu(x))/d(x) = 0` if `x <= 0`

`float(self.data > 0)` evaluates the boolean comparison and converts it to `1.0` or `0.0`. When `x` is exactly `0`, the gradient is `0` (subgradient convention).

#### Convenience aliases (lines 51–57)

```python
def __neg__(self): return self * -1
def __radd__(self, other): return self + other
def __sub__(self, other): return self + (-other)
def __rsub__(self, other): return other + (-self)
def __rmul__(self, other): return self * other
def __truediv__(self, other): return self * other**-1
def __rtruediv__(self, other): return other * self**-1
```

These are not new primitive operations; they compose the primitives above. Subtraction is addition of a negation. Division is multiplication by the inverse (`x^-1`). This means the computation graph for `a / b` is actually three nodes: `b^(-1)`, then `a * (b^(-1))`.

---

## The backward pass

### Topological sort (lines 59–68)

```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)
    build_topo(self)
```

Before applying the chain rule, you need to visit nodes in an order that guarantees: a node's gradient is fully accumulated before that gradient is propagated to its children.

That ordering is the **reverse of a topological sort** of the DAG. `build_topo` is a standard depth-first post-order traversal: it recurses into all children before appending the current node. The result is that leaf nodes (inputs) appear first in `topo`, and the root (loss) appears last.

```
For c = a*b + a, build_topo(c) produces:

  Recurse into (a*b): recurse into a, recurse into b
    → append a, append b, append (a*b)
  Recurse into a (already visited, skip)
  → append c

  topo = [a, b, (a*b), c]

  reversed(topo) = [c, (a*b), a, b]
                    ↑ process in this order during backward
```

### Chain rule application (lines 69–72)

```python
    self.grad = 1
    for v in reversed(topo):
        for child, local_grad in zip(v._children, v._local_grads):
            child.grad += local_grad * v.grad
```

**`self.grad = 1`**: The gradient of the output with respect to itself is always 1 — a scalar's derivative with respect to itself is 1 by definition. This is the seed that starts the backward pass.

**`child.grad += local_grad * v.grad`**: This is the chain rule in its most mechanical form.

The chain rule states: if `c = f(a)` and `loss = g(c)`, then:

```
d(loss)/d(a) = d(loss)/d(c) * d(c)/d(a)
                    ↑                ↑
                 v.grad          local_grad
```

`v.grad` is the gradient that has already been accumulated for node `v` (how much the loss changes when `v` changes). `local_grad` is how much `v` changes when the child changes. Their product is how much the loss changes when the child changes.

**Why `+=` and not `=`?** Because a single node can appear as a child of multiple parent nodes. If `a` feeds into both `a*b` and the second branch of `a*b + a`, then `a.grad` receives contributions from both parents. Using `+=` accumulates all of them; using `=` would overwrite previous contributions and give wrong results.

```
Gradient flow for c = a*b + a  (a=2, b=3, c=8):

  Start: c.grad = 1

  Process c (children: a*b, a):
    (a*b).grad += 1 * 1 = 1
    a.grad     += 1 * 1 = 1

  Process (a*b) (children: a, b):
    a.grad += 3 * 1 = 3   ← local_grad for a in a*b is b.data = 3
    b.grad += 2 * 1 = 2   ← local_grad for b in a*b is a.data = 2

  Final:
    a.grad = 1 + 3 = 4    (correct: dc/da = b + 1 = 3 + 1 = 4)
    b.grad = 2             (correct: dc/db = a = 2)
```

---

## Hands-on REPL exercises

Open a Python REPL in your project directory and paste the imports:

```python
import math
import sys
sys.path.insert(0, '.')
# Paste the Value class here, or load it:
exec(open('microgpt.py').read().split('# Initialize')[0])  # loads only the Value class
```

A cleaner approach for a scratch session — paste just the class definition (lines 30–72 of `microgpt.py`) directly into the REPL.

---

### Exercise 1: Basic multiply and backward

```python
a = Value(3.0)
b = Value(4.0)
c = a * b      # c.data == 12.0

c.backward()

print(a.grad)  # Expected: 4.0  (dc/da = b = 4)
print(b.grad)  # Expected: 3.0  (dc/db = a = 3)
```

> **What's happening:** `c._local_grads` is `(4.0, 3.0)` — the values of `b` and `a` at construction time. `backward()` seeds `c.grad = 1`, then multiplies each local grad by `c.grad` and accumulates into `a.grad` and `b.grad`.

---

### Exercise 2: Polynomial f(x) = x^2 + 2x + 1

The derivative of `f(x) = x^2 + 2x + 1` is `f'(x) = 2x + 2`. At `x = 3`, `f'(3) = 8`.

```python
x = Value(3.0)
f = x**2 + 2*x + 1

f.backward()

print(f.data)  # Expected: 16.0  (9 + 6 + 1)
print(x.grad)  # Expected: 8.0   (2*3 + 2)
```

> **What's happening:** `x` appears three times in the graph (once in `x**2`, once in `2*x`, once nowhere — the constant `1` has no `x` child). The `+=` in gradient accumulation ensures that contributions from `x**2` and `2*x` are both added to `x.grad`.

Verify that the graph accumulates correctly by printing intermediate nodes:

```python
x = Value(3.0)
t1 = x**2        # t1 = 9
t2 = 2 * x       # t2 = 6
t3 = t1 + t2     # t3 = 15
f  = t3 + 1      # f  = 16

f.backward()

print(t1.grad)   # Expected: 1.0
print(t2.grad)   # Expected: 1.0
print(x.grad)    # Expected: 8.0  (comes from both t1 and t2 branches)
```

---

### Exercise 3: Diamond dependency — the `+=` test

This is the example traced earlier: `a=2`, `b=3`, `c = a*b + a`.

```python
a = Value(2.0)
b = Value(3.0)

mul = a * b      # mul.data = 6.0
c   = mul + a    # c.data   = 8.0

c.backward()

print(a.grad)    # Expected: 4.0  (b + 1 = 3 + 1)
print(b.grad)    # Expected: 2.0  (a = 2)
```

> **What's happening:** `a` is a child of both `mul` and `c`. When `backward()` processes `c`, it gives `a.grad += 1 * 1 = 1`. When it processes `mul`, it gives `a.grad += 3 * 1 = 3`. Total: `a.grad = 4`. If you replaced `+=` with `=`, the second assignment would overwrite the first and you'd get `a.grad = 3` — wrong.

To see this break intentionally, try patching the backward loop temporarily:

```python
# Demonstration only — do NOT use this in real code
def broken_backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)
    build_topo(self)
    self.grad = 1
    for v in reversed(topo):
        for child, local_grad in zip(v._children, v._local_grads):
            child.grad = local_grad * v.grad   # BUG: = instead of +=

a = Value(2.0)
b = Value(3.0)
c = a * b + a

broken_backward(c)
print(a.grad)    # Prints 3.0 — wrong! Should be 4.0
```

---

## Eager vs lazy local gradients

Notice that `_local_grads` is computed **during the forward pass**, at the moment you call the operation:

```python
def __mul__(self, other):
    return Value(self.data * other.data, (self, other), (other.data, self.data))
    #                                                     ↑ captured right now
```

The values `other.data` and `self.data` are read at this moment and stored as plain Python floats. They do not track further changes to `self` or `other`. This is fine because in a standard forward pass the `.data` of a node never changes after creation — the graph is built once, then differentiated once.

A lazy design would instead store a closure (a function) that recomputes `other.data` when called during the backward pass. That would be more flexible but would add a function call overhead for every edge in the graph on every backward pass. The eager approach is simpler and faster for this use case.

---

## Check your understanding

**1.** What are the `_local_grads` for the expression `out = a.exp()` when `a.data = 2.0`? Write them out as a tuple before running the code, then verify.

**2.** In the topological sort, why must a node be appended to `topo` **after** recursing into all its children, rather than before? What would go wrong if the append came first?

**3.** Suppose you compute `y = x * x` (using two separate multiplications rather than `x**2`). How many times does `x.grad` get updated during `y.backward()`? What is the final value of `x.grad` when `x.data = 5.0`? (Hint: think about how many times `x` appears as a child in the graph.)

---

## What's next

Chapter 4 walks through the model parameters and the forward pass: how `Value` nodes are organized into weight matrices, how `linear`, `rmsnorm`, `softmax`, and the attention mechanism chain these nodes together into a full GPT computation graph, and why the graph can reach tens of thousands of nodes even for the tiny configuration in `microgpt.py`.

By the end of Chapter 4 you will be able to read the entire forward pass and reason about what the computation graph looks like before `loss.backward()` is called.
