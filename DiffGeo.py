# This module implements frameworks of differential geometry.
from EinIndex import *
import sympy


def christoffel(simp=True):
    i, m, k, l = Index.new("i m k l")  # This function produces pure tensors, so creating indices is safe
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
        temp_ind = Index.new("temp_index_in_covariant_derivative")[0]  # No index name is gonna crash that one, right??
        T = Tensor(ind=(*op.tensor.IndexType, COV))
        T[(*op.ind, self.index)] = D(self.index)(op)
        for i in range(len(op.ind)):  # messy code... TODO repair this?
            indl = op.ind[:]
            indl[i] = temp_ind if op.ind[i].contrv else -temp_ind
            T[(*op.ind, self.index)] += self.connection[op.ind[i], self.index, -temp_ind] * op.tensor[indl]\
                if op.ind[i].contrv else -self.connection[temp_ind, self.index, op.ind[i]] * op.tensor[indl]
        return T[(*op.ind, self.index)]


def riemann(simp=True):
    pass


if __name__ == "__main__":
    x, y = x[0], x[1]
    # Spherical metric: ds^2 = dy^2 + sin^2 y dx^2
    Tensor.METRIC = Tensor(arr=np.array([[sympy.sin(y) ** 2, Real(0)], [Real(0), Real(1)]]), ind=(COV, COV))
    Tensor.INVERSE_METRIC = Tensor(arr=np.array([[1 / sympy.sin(y) ** 2, Real(0)], [Real(0), Real(1)]]),
                                   ind=(CONTRAV, CONTRAV))
    a,b,c,d = Index.new('a b c d')
    print(Tensor.METRIC[-a, -b] * Tensor.INVERSE_METRIC[b, c])  # Metric is normalizing correctly
    print()
    Nabla.connection = christoffel()  # This test is based on Carrol(2003). The result is consistent.
    Nabla.connection.trig_simp()
    print(Nabla.connection)
    T = Tensor((COV, COV, COV))
    T[-a,-b,-c] = Nabla(-a)(Tensor.METRIC[-b, -c])
    T.trig_simp()
    print(T)  # Metric compatibility! Hurray!
