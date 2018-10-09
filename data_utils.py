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

import jsonpickle as json
class InstanceData:
    """
    Represents a problem's instance data
    """

    def __init__(self, clients, locations):
        self.c_num = clients
        self.l_num = locations
        self.min = 0
        self.max = 10

        # generate data        
        self.U = np.zeros(self.c_num)        
        self.C = np.random.randint(self.min, self.max+1, (self.c_num, self.l_num))
        avg_cp = [np.average(self.C[i]) for i in range(self.c_num)]
        #print("avg_cp = \n", avg_cp)
        self.F = [np.random.randint(avg_cp[i]/2, avg_cp[i]) for i in range(self.c_num)]  
        
        
    def save(self):
        # serialize instance
        data = json.encode(self)
        # create file
        name = "instance_{}_{}.json".format(self.c_num, self.l_num)
        file = open(name, "w+")
        # write data
        file.write(data)
        # close file
        file.close()

    def load(self):
        # read file
        name = "instance_{}_{}.json".format(self.c_num, self.l_num)
        file = open(name, "r")
        # deserialize instance
        data = file.read()
        instance = json.decode(data)
        return instance

    def print(self):
        print("U = \n", self.U)
        print("C = \n", self.C)
        print("F = \n", self.F)