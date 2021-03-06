import numpy as np
from pulp import LpBinary, LpInteger, LpContinuous, LpVariable

class DataGenerator:
    """
    Data generator class
    """

    def INT_vector(self, len=10, min=0, max=10, to_print=False):
        v = np.random.randint(low=min, high=max+1, size=len)
        if to_print:
            print(f"INT_vector : {v}")
        return v

    def INT_matrix(self, rows=10, columns=10, min=0, max=10, to_print=False):
        m = np.random.randint(min, max+1, (rows, columns))
        if to_print:
            print(f"INT_matrix : \n{m}")
        return m

    def VAR_vector(self, size=10, prefix='x', type=LpBinary, to_print=False):
        v = LpVariable.dicts(prefix, range(size), cat=type, lowBound=0)
        if to_print:
            print(f"VAR_vector : {v} - ({type.lower()})")
        return v

    def VAR_matrix(self, rows=10, columns=10, prefix='x', type=LpBinary, to_print=False):    
        m = LpVariable.dicts(prefix, (range(rows), range(columns)), cat=type, lowBound=0)
        if to_print:
            print(f"VAR_matrix : \n{m} - ({type.lower()})")
        return m