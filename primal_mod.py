from solver_utils import *
from data_utils import *
from pulp import *

dg = DataGenerator()

def build_IP(clients, locations, c_matrix, f_vect):
    # problem instance
    problem = LpProblem("UFL - Primal Problem")

    # get cost values
    # C = build_C_matrix(clients, locations) # service costs matrix
    # F = build_F_vector(locations) # fixed costs vector

    # decision variables
    x = dg.VAR_matrix(clients, locations, 'x', 'Binary')
    y = dg.VAR_vector(locations, 'y', 'Binary')

    # objective function
    service_costs = [c_matrix[i][j] * x[i][j] for i in range(clients) for j in range(locations)]
    fixed_costs = [f_vect[j] * y[j] for j in range(locations)]
    problem += lpSum(service_costs) + lpSum(fixed_costs), "Total Costs"

    # constraints
    for i in range(clients):
        problem += lpSum([x[i][j] for j in range(locations)]) == 1
    
    for i in range(clients):
        for j in range(locations):
            problem += x[i][j] - y[j] <= 0

    return problem


if __name__ == '__main__':
    clients = 10
    locations = 5000
    c_matrix = dg.INT_matrix(clients, locations)
    f_vector = dg.INT_vector(locations)

    # build IP problem
    p_ip = build_IP(clients, locations, c_matrix, f_vector)

    # solve IP problem
    solve(p_ip)
    # print_variables(p_ip)     
    v_ip = get_objectiveFunction_value(p_ip)
    print(v_ip)
    print_all_variables(p_ip)