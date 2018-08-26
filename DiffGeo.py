# This module implements frameworks of differential geometry.
from EinIndex import *
import sympy


def christoffel(simp=True):
    i, m, k, l = Index.new("i m k l")
    C = Tensor(ind=(CONTRAV, COV, COV))
    C[i, -k, -l] = Tensor.INVERSE_METRIC[i, m] * (
                D(-k)(Tensor.METRIC[-l, -m]) + D(-m)(Tensor.METRIC[-k, -l]) + D(-l)(Tensor.METRIC[-m, -k])) / 2.
    if simp:
        C.simplify()
    return C


class Nabla:
    """Implements covariant derivative."""
    pass


def riemann(simp=True):
    pass

if __name__ == "__main__":
    x, y = x[0], x[1]
    # Spherical metric: ds^2 = dy^2 + sin^2 y dx^2
    Tensor.METRIC = Tensor(arr=np.array([[sympy.sin(y) ** 2, Real(0)], [Real(0), Real(1)]]), ind=(COV, COV))
    Tensor.INVERSE_METRIC = Tensor(arr=np.array([[1 / sympy.sin(y) ** 2, Real(0)], [Real(0), Real(1)]]),
                                   ind=(CONTRAV, CONTRAV))
    print(christoffel())
