import matplotlib.pyplot as plt
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
        v = LpVariable.dicts(prefix, range(size), cat=type)
        if to_print:
            print(f"VAR_vector : {v} - ({type.lower()})")
        return v

    def VAR_matrix(self, rows=10, columns=10, prefix='x', type=LpBinary, to_print=False):    
        m = LpVariable.dicts(prefix, (range(rows), range(columns)), cat=type)
        if to_print:
            print(f"VAR_matrix : \n{m} - ({type.lower()})")
        return m

class DataLogger:
    """
    Data logger class
    """

    d = dict()

    def logData(self, key, data):
        if key not in self.d:            
            self.d[key] = list()
        self.d[key].append(data)

    def plotData(self, key, title):
        plt.plot(self.d[key])
        plt.title(title)        
        #plt.show()

    def plotSubgradientProcess(self, lr_key, ip_key):
        lr_data = self.d[lr_key]        
        ip_data = self.d[ip_key] * len(lr_data)
        plt.plot(lr_data)
        plt.plot(ip_data, '--')
        plt.title("Subgradient process")    
        plt.ylabel("Objective function value")
        #plt.show()