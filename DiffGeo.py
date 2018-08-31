# This module implements frameworks of differential geometry.
from EinIndex import *
import sympy

temp_ind = Index.new("temp_index_in_DiffGeo.py")[0]  # No index name is gonna crash that one, right??
i, m, k, l = Index.new("i m k l")  # This function processes pure tensors, so creating indices is safe


def christoffel(simp=True):
    """Returns a tensor(awww) Gamma[i, -k, -l] which is the Christoffel symbol associated with the metric."""
    C = Tensor(ind=(CONTRAV, COV, COV))
    C[i, -k, -l] = Tensor.INVERSE_METRIC[i, m] * (
            D(-k)(Tensor.METRIC[-l, -m]) - D(-m)(Tensor.METRIC[-k, -l]) + D(-l)(Tensor.METRIC[-m, -k])) / 2.
    if simp:
        C.simplify()
    return C


class Nabla:
    """Implements covariant derivative."""
    connection = christoffel()

    def __init__(self, index):
        if index.contrv and not Tensor.IMPLICIT_INDEX_RAISING_LOWERING:
            raise ValueError('The derivative must be covariant.')
        elif index.contrv:
            raise NotImplementedError('Not Implemented: implicit index raising/lowering')
        self.index = index

    def __call__(self, op: IndexHandle):
        T = Tensor(ind=(*op.tensor.IndexType, COV))
        T[(*op.ind, self.index)] = D(self.index)(op)
        for i in range(len(op.ind)):  # messy code... TODO repair this?
            indl = op.ind[:]
            indl[i] = temp_ind if op.ind[i].contrv else -temp_ind
            T[(*op.ind, self.index)] += self.connection[op.ind[i], self.index, -temp_ind] * op.tensor[indl]\
                if op.ind[i].contrv else -self.connection[temp_ind, self.index, op.ind[i]] * op.tensor[indl]
        return T[(*op.ind, self.index)]


def commutator(X: Tensor, Y: Tensor, j:Index) -> Tensor:
    """Returns the commutator [X,Y] of two vector fields."""
    return (D(temp_ind)(X[temp_ind])*Y[j] - D(temp_ind)(Y[temp_ind])*X[j]).tensor


def riemann(connection=None, simp=True):
    """Returns the Riemann curvature tensor.
    Warning: the convention used here is the same as Carrol(1997) at
    http://ned.ipac.caltech.edu/level5/March01/Carroll3/Carroll3.html"""
    if connection is None:
        connection = christoffel()
    R = Tensor(ind=(CONTRAV, COV, COV, COV))
    R[i, -m, -k, -l] = D(-k)(connection[i, -l, -m]) - D(-l)(connection[i, -k, -m]) + \
        connection[i, -k, -temp_ind] * connection[temp_ind, -l, -m] - \
        connection[i, -l, -temp_ind] * connection[temp_ind, -k, -m]
    return R


if __name__ == "__main__":
    x, y = x[0], x[1]
    # Spherical metric: ds^2 = dy^2 + sin^2 y dx^2
    Tensor.METRIC = Tensor(arr=np.array([[sympy.sin(y) ** 2, Real(0)], [Real(0), Real(1)]]), ind=(COV, COV))
    Tensor.INVERSE_METRIC = Tensor(arr=np.array([[1 / sympy.sin(y) ** 2, Real(0)], [Real(0), Real(1)]]),
                                   ind=(CONTRAV, CONTRAV))
    a,b,c,d = Index.new('a b c d')
    print(Tensor.METRIC[-a, -b] * Tensor.INVERSE_METRIC[b, c])  # Metric is normalizing correctly
    print()
    Nabla.connection = christoffel()  # This test is based on Carrol(1997). The result is consistent.
    print(Nabla.connection)
    T = Tensor((COV, COV, COV))
    T[-a,-b,-c] = Nabla(-a)(Tensor.METRIC[-b, -c])
    T.expand()
    T.trig_simp()
    print("T", T)  # Metric compatibility! Hurray!
    R = riemann()
    R.trig_simp()
    R.simplify()
    print(R)
    v = R[a, -b, -a, -d] * Tensor.INVERSE_METRIC[b, d]
    print(v)  # TODO Here is a mysterious calculation error, which I cannot find the source...
