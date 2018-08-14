import numpy as np
import sympy
from sympy import RealNumber as Real, symbols as sym, diff as d

DIMENSION = 2
x = sym('x y')

CONTRAV = True
COV = False


class Index:
    def __init__(self, name):
        self.name = name.strip('-')
        self.contrv = (name[0] != '-')

    @staticmethod
    def new(names):
        return tuple([Index(i) for i in names.split()])

    def __repr__(self):
        return self.name if self.contrv else '-' + self.name

    def __neg__(self):
        if self.contrv:
            return Index('-' + self.name)
        else:
            return Index(self.name)

    def __eq__(self, other):
        return self.name == other.name and self.contrv == other.contrv


class Coord:
    """
    Manages coordinate transforms between different sets of indices.
    Usage: create transformations by tr = Coord(...), then use tr(index).
    """
    pass  # TODO


class IndexHandle:
    """
    The art of representing a whole tensor with a component...
      comprising coordinate transform, internal spaces etc.
    Tensor T := T_{ij}^{mn}
    In PyGR as T[-i,-j,m,n],where i/j/m/n is of type IndexHandle.
    Contraction: T_{ij}*U^{j}
     as contr(T[-i,-j]*U[j],j)? To avoid certain ambiguities...
    """

    def __init__(self, T, i, do_not_contract=False):
        """
        Creates an index handle
        :param T: Tensor handled
        :param i: tuple of indices
        """
        self.ind = i
        # Contraction
        if not do_not_contract:
            rten = T.T
            itype = []
            icopy = []
            i = list(i)
            inum = 0
            while True:
                curind = i[inum]
                try:
                    cind = i.index(-curind, inum + 1)
                    rten = np.sum(np.diagonal(rten, axis1=inum, axis2=cind), axis=-1)
                    i.remove(curind)
                    i.remove(-curind)
                except ValueError:
                    itype.append(curind.contrv)
                    icopy.append(curind)
                    inum += 1
                finally:
                    if inum >= len(i):
                        break
            self.ind = icopy
            self.tensor = Tensor(ind=tuple(itype), arr=rten.copy())
        else:
            self.tensor = T

    def __add__(self, other):
        if self.ind != other.ind:
            raise ValueError(f'Indices do not match: {self.ind} and {other.ind}')
        else:
            return IndexHandle(self.tensor + other.tensor, self.ind)

    def __mul__(self, other):
        if type(other) == float:
            return IndexHandle(self.tensor * other, self.ind)
        return IndexHandle(self.tensor * other.tensor, (*self.ind, *other.ind))

    __rmul__ = __mul__
    __radd__ = __add__

    def __repr__(self):
        return repr(self.tensor)

    def __sub__(self, other):
        if self.ind != other.ind:
            raise ValueError(f'Indices do not match: {self.ind} and {other.ind}')
        else:
            return IndexHandle(self.tensor - other.tensor, self.ind)

    def __truediv__(self, other):
        if type(other) == float:
            return IndexHandle(self.tensor / other, self.ind)
        return IndexHandle(self.tensor / other.tensor, (*self.ind, *[-i for i in other.ind]))

    def __rsub__(self, other):
        if self.ind != other.ind:
            raise ValueError(f'Indices do not match: {self.ind} and {other.ind}')
        else:
            return IndexHandle(other.tensor - self.tensor, self.ind)

    def __rtruediv__(self, other):
        if type(other) == float:
            return IndexHandle(other / self.tensor, self.ind)
        return IndexHandle(other.tensor / self.tensor, (*other.ind, *[-i for i in self.ind]))


vec_exp = np.frompyfunc(sympy.expand, 1, 1)
vec_sym = np.frompyfunc(sympy.simplify, 1, 1)


class Tensor:
    def __init__(self, ind, arr=None):
        if type(ind) == Index:
            ind = (ind,)
        if arr is None:
            self.T = np.zeros((DIMENSION,) * len(ind), dtype=Real)
        else:
            self.T = arr.reshape((DIMENSION,) * len(ind))
        self.IndexType = tuple(ind)  # Should be like (True, False,...);True for contravariant
        self.expand()

    def __getitem__(self, index):
        if type(index) == Index:
            index = (index,)
        it = tuple([i.contrv for i in index])
        if self.IndexType != it:
            raise ValueError('Index type does not match.')
        return IndexHandle(self, index)

    def __setitem__(self, ind, other: IndexHandle):
        axes = []
        for i in ind:
            axes.append(other.ind.index(i))
        self.T = other.tensor.T.transpose(axes)
        self.expand()

    def __add__(self, other):
        if self.IndexType != other.IndexType:
            raise ValueError('Index type does not match.')
        return Tensor(arr=self.T + other.T, ind=self.IndexType)

    def __mul__(self, other):
        if type(other) == float:
            return Tensor(arr=self.T * other, ind=self.IndexType)
        return Tensor(arr=np.tensordot(self.T, other.T, axes=0), ind=(*self.IndexType, *other.IndexType))

    def __sub__(self, other):
        if self.IndexType != other.IndexType:
            raise ValueError('Index type does not match.')
        return Tensor(arr=self.T - other.T, ind=self.IndexType)

    def __neg__(self):
        return Tensor(arr=-self.T, ind=self.IndexType)

    def __truediv__(self, other):
        if type(other) == float:
            return Tensor(arr=self.T / other, ind=self.IndexType)
        return Tensor(arr=np.tensordot(other.T, 1 / self.T, axes=0),
                      ind=(*other.IndexType, *[not i for i in self.IndexType]))

    __rmul__ = __mul__
    __radd__ = __add__

    def __rsub__(self, other):
        if self.IndexType != other.IndexType:
            raise ValueError('Index type does not match.')
        return Tensor(arr=other.T - self.T, ind=self.IndexType)

    def __rtruediv__(self, other):
        if type(other) == float:
            return Tensor(arr=other / self.T, ind=self.IndexType)
        return Tensor(arr=np.tensordot(self.T, 1 / other.T, axes=0), ind=(*self.IndexType,
                                                                          *[not i for i in other.IndexType]))

    def __repr__(self):
        return str(self.T)

    def simplify(self):
        self.T = vec_sym(self.T)

    def expand(self):
        self.T = vec_exp(self.T)


class D:
    """Implements derivative.
    Usage: D[index]()
    """
    pass


G = Tensor(arr=np.array([[Real(1), 0], [1, 0]]), ind=(COV, COV))

if __name__ == '__main__':
    # UNIT TEST
    arr = np.array([x[0], x[1]])
    V: Tensor = Tensor(arr=arr, ind=(CONTRAV,))
    a, b, c, d, e, f = Index.new('a b c d e f')
    print(V[a] * V[b], V[a] + V[a], V[a] - V[a], V[b] / V[a], 0.3 / V[a], sep='\n')
    R = Tensor(arr=np.array([[[1, 0], [0, x[0]]], [[x[1] + 1, 0], [x[0] * x[1] - 2, 0]]]), ind=(CONTRAV, CONTRAV, COV))
    V = Tensor(ind=(CONTRAV, CONTRAV, CONTRAV, COV))
    print(V)
    V[a, c, d, -e] = R[a, b, -e] * R[c, d, -b]
    print(V)
    print(D[3])
