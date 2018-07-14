import numpy as np
from sympy import RealNumber as RR, symbols as sym, diff as D

DIMENSION = 2


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


class IndexHandle:
    """
    The art of representing a whole tensor with a component...
      comprising coordinate transform, internal spaces etc.
    Tensor T := T_{ij}^{mn}
    In PyGR as T[-i,-j,m,n],where i/j/m/n is of type IndexHandle.
    Contraction: T_{ij}*U^{j}
     as contr(T[-i,-j]*U[j],j)? To avoid certain ambiguities...
    """

    def __init__(self, T, i):
        """
        Creates an index handle
        :param T: Tensor handled
        :param i: tuple of indices
        """
        self._tensor = T
        self._ind = i

    def __add__(self, other):
        if self._ind != other._ind:
            raise ValueError(f'Index types do not match: {self._ind} and {other._ind}')
        else:
            return IndexHandle(self._tensor + other._tensor, self._ind)


class Tensor:
    def __init__(self, arr, ind):
        self.T = arr.reshape((DIMENSION,) * len(ind))
        self.IndexType = tuple(ind)  # Should be like (True, False,...);True for contravariant

    def __getitem__(self, index):
        it = tuple([i.contrv for i in index])
        if self.IndexType != it:
            raise ValueError('Index type does not match.')
        return IndexHandle(self, index)

    def __add__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __sub__(self, other):
        pass

    def __truediv__(self, other):
        pass


if __name__ == '__main__':
    a, cb = Index.new('a -b')
    print(a, cb, -cb)
    arr = np.array([sym('x'), sym('y')])
    V = Tensor(arr=arr, ind=(True,))
    Va = V[a,]
    Vb = V[-cb,]
    aV = V[a,] + V[a,]
