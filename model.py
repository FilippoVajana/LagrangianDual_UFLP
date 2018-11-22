from pulp import *
from data_utils import DataGenerator

def build_primal(c_num, l_num, C, F):
    '''
    Builds primal model
    '''
    data = DataGenerator()
    # problem instance
    problem = LpProblem("UFL - Primal Problem (Wolsey)", sense=LpMaximize)

    # decision variables
    x = data.VAR_matrix(c_num, l_num, 'x', 'Continuous')
    y = data.VAR_vector(l_num, 'y', 'Binary')

    # objective function
    profits = [C[i][j] * x[i][j] for i in range(c_num) for j in range(l_num)]
    fixed_costs = [F[j] * y[j] for j in range(l_num)]
    problem += lpSum(profits) - lpSum(fixed_costs)

    # constraints
    for i in range(c_num):
        problem += lpSum([x[i][j] for j in range(l_num)]) == 1

    for i in range(c_num):
        for j in range(l_num):
            problem += x[i][j] - y[j] <= 0
            
    return problem

def build_relaxed(c_num, l_num, C, F):
    '''
    Builds integer relaxed model
    '''
    data = DataGenerator()
    # problem instance
    problem = LpProblem("UFL - Primal Integer Relaxed Problem (Wolsey)", sense=LpMaximize)

    # decision variables
    x = data.VAR_matrix(c_num, l_num, 'x', 'Continuous')
    y = data.VAR_vector(l_num, 'y', 'Continuous')

    # objective function
    profits = [C[i][j] * x[i][j] for i in range(c_num) for j in range(l_num)]
    fixed_costs = [F[j] * y[j] for j in range(l_num)]
    problem += lpSum(profits) - lpSum(fixed_costs)

    # constraints
    for i in range(c_num):
        problem += lpSum([x[i][j] for j in range(l_num)]) == 1

    for i in range(c_num):
        for j in range(l_num):
            problem += x[i][j] - y[j] <= 0
            
    return problem

def build_lagrange(c_num, l_num, u_vector, C, F):
    """
    Lagrange relaxation for UFL problem as in Wolsey
    """
    data = DataGenerator()
    # problem instance
    problem = LpProblem("UFL - Lagrange Relaxed Problem (Wolsey)", sense=LpMaximize)

    # check lagrange multipliers vector
    if len(u_vector) != c_num : 
        raise ValueError('Invalid u_vect size')

    # decision variables    
    x = data.VAR_matrix(c_num, l_num, 'x', 'Continuous')
    y = data.VAR_vector(l_num, 'y', 'Binary')

    # objective function
    profits = [C[i][j] * x[i][j] for i in range(c_num) for j in range(l_num)]
    fixed_costs = [F[j] * y[j] for j in range(l_num)]
    l1 = [u_vector[i] for i in range(c_num)]
    l2 = [u_vector[i] * x[i][j] for i in range(c_num) for j in range(l_num)]

    problem += lpSum(profits) - lpSum(fixed_costs) + lpSum(l1) - lpSum(l2)

    # constraints
    for i in range(c_num):
        for j in range(l_num):
            problem += x[i][j] - y[j] <= 0

    return problem