import numpy as np
from sympy import RealNumber as RR, symbols as sym, diff as D

DIMENSION = 2

class Index:
    def __init__(self,name):
        self.name = name.strip('-')
        self.contrv = (name[0] != '-')
        
    @staticmethod
    def new(names):
        return tuple([Index(i) for i in names.split()])

    def __repr__(self):
        return self.name if self.contrv else '-' + self.name

    def __neg__(self):
        if self.contrv:
            return Index('-'+self.name)
        else:
            return Index(self.name)
    
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
        pass

    
    

class Tensor:
    def __init__(self, arr, ind):
        self.T = arr.reshape((DIMENSION,)*len(ind))
        self.IndexType = tuple(ind)     #Should be like (True, False,...);True for contravariant
    def __getitem__(self,index):
        return IndexHandle(self,index)

if __name__ == '__main__':
    a, cb = Index.new('a -b')
    print(a,cb,-cb)
