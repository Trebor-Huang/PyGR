# PyGR
Implements symbolic calculation in General Relativity. I started this to enhance my understanding of GR, and to improve my programming skills (as a physics/math student, of course).

I haven't bothered to do PyPI or anything yet, but this package depends on `numpy` and `sympy`. Below is the current structure of this project.

## EinIndex.py
Implements the [Einstein summation convention](https://en.wikipedia.org/wiki/Einstein_notation).

The class `IndexHandle` does all the useful job. You can manipulate tensors in the convenient index notation, such as

    a, b, c, d = Index.new("a b c d")  # Similar style with sympy.symbols
    V[a, b] = T[a, b] + R[b, a]  # V, T and R are predeclared (0,2)-rank tensors
    P[a, b, c, d] = T[a, b] * R[d, c]  # P is predeclared as (0,4)
    C[a, b] = T[a, d] * S[-d, b]  # S is a (1,1)-tensor. This does the automatic contraction. Here the minus sign means a conversion between covariant lower indices and contravariant upper indices
    
However, you have to declare tensors first (it's the best syntax sugar I can do, and possibly also Python's best). You can do this by:

    V = Tensor(ind=(CONTRAV, CONTRAV))  # Create empty tensor
    S = Tensor(arr=np.array([[0,1],[x[0],x[1]]], ind=(COV, CONTRAV))  # Create tensor with specified value. x is a predefined list of coordinate symbols.
    
The class `D` is for (ordinary partial) differentiation.

    Q[a, b, -c] = D(-c)(V[a,b])  # Notice the funny notation
    
Currently, the default manifold is 2D Euclidean plane. You can change this by modifying `DIMENSION` and `Tensor.METRIC` and `Tensor.INVERSE_METRIC` constants in this module.

## DiffGeo.py
Implements some important objects of differential geometry.
