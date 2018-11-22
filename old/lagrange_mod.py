from solver_utils import *
from data_utils import *
from primal_mod import *
from pulp import *

dg = DataGenerator()

def build_LR(clients, locations, c_matrix, f_vect, u_vect):
    """
    Lagrange relaxation removing client demand constraints
    """
    # problem instance
    problem = LpProblem("UFL - Lagrange Relaxed Problem")

    # check lagrange multipliers vector
    if len(u_vect) != clients : 
        raise ValueError('Invalid u_vect size')

    # decision variables
    x = dg.VAR_matrix(clients, locations, 'x', 'Binary')
    y = dg.VAR_vector(locations, 'y', 'Binary')

    # objective function
    s_c = [c_matrix[i][j] * x[i][j] for i in range(clients) for j in range(locations)]
    f_c = [f_vector[j] * y[j] for j in range(locations)]
    lag_mult = [u_vect[i] * (1 - x[i][j]) for i in range(clients) for j in range(locations)]

    problem += lpSum(s_c) + lpSum(f_c) + lpSum(lag_mult)

    # constraints
    for i in range(clients):
        for j in range(locations):
            problem += x[i][j] - y[j] <= 0

    return problem


def compute_subgradient(size, x_lr):
    x_vect = np.array([np.sum(x_lr) for r in range(size)])
    sg = 1 - x_vect
    return sg

def compute_step(alpha, subgradient, z_ip, z_lr):
    z_diff = np.abs(z_ip - z_lr)
    #print(f"z_diff = {z_diff}")

    # s_norm = np.linalg.norm(subgradient)
    # s_norm2 = np.power(s_norm, 2, dtype = float)
    #print(f"s_norm2 = {s_norm2}")

    subgrad2 = subgradient ** 2
    subgrad2_sum = np.sum(subgrad2)
    #print(f"subgrad2_sum = {subgrad2_sum}")
    
    step = alpha * (z_diff) / subgrad2_sum
    return step

def update_lagrange(u, step, subgradient):
    u_t = u + step * subgradient
    u_next = [max(0, u_t[i]) for i in range(u.size)]
    return np.array(u_next)
    
def check_end_conditions(z, z_ub, alpha):
    if z == z_ub: return True
    if alpha < 0.005: return True
    return False

if __name__ == '__main__':
    '''
    Subgradient Method:
    Preprocessing) solve IP problem
    1) build LR(u) problem
    2) solve LR(u)
    3) compute subgradient S = d - D*x
    4) check for S = 0
    5) compute step size
    6) update u parameter
    7) update best solution
    8) check stop conditions
    '''
    #init
    clients = 10
    locations = 4
    c_matrix = dg.INT_matrix(clients, locations, 1, 10)
    f_vector = dg.INT_vector(locations, 10, 100)    
    u_vector = np.ones(clients)
    logger = DataLogger()

    iteration = 1
    alpha = float(2)
    z_max = -np.inf

    # preprocessing
    p_ip = build_IP(clients, locations, c_matrix, f_vector)
    solve(p_ip) #solve integer problem

    v_ip = get_objectiveFunction_value(p_ip)
    logger.logData("z_ip", v_ip)
    print(f"v_ip:\n{v_ip}\n")
    x_ip = get_variable_value(p_ip, 'x') 
    print(f"x_ip:\n{x_ip.reshape((clients, locations))}\n")
    
    #subgradient techinque
    while True:

        #step 1
        p_lr = build_LR(clients, locations, c_matrix, f_vector, u_vector)

        #step 2
        solve(p_lr)
        logger.logData("z_lr", get_objectiveFunction_value(p_lr))
        x_lr = get_variable_value(p_lr, 'x')
        #print(f"x_lr:\n{x_lr}\n")

        #step 3    
        subgrad = compute_subgradient(clients, x_lr)
        #print(f"subgradient:\n{subgrad}\n")

        #step 4
        if subgrad.any() == False:
            print("Subgradient stop condition")
            print(f"\tResult = {x_lr}")
            break

        #step 5
        z_ip = get_objectiveFunction_value(p_ip)
        z_lr = get_objectiveFunction_value(p_lr)
        step = compute_step(alpha, subgrad, z_ip, z_lr)
        #print(f"step:\n{step}\n")
        #print(f"z_lr = {z_lr}")

        #step 6
        u_vector = update_lagrange(u_vector, step, subgrad)
        #print(f"u_vector = \n{u_vector}")
        
        #step 7
        if  z_lr > z_max: 
                z_max = z_lr
                print(f"Updated z_max = {z_max} after {iteration} iterations")
                iteration = 0            
        
        if iteration == 30:
            alpha = alpha / 2
            print(f"alpha:\n{alpha}\n")
            iteration = 0

        #step 8
        iteration = iteration + 1
        end = check_end_conditions(z_max, z_ip, alpha)
        if end: 
            print(f"Objective function value = {z_max}")
            print(f"X = \n{x_lr.reshape((clients, locations))}")
            print(f"Lagrange multipliers = \n{u_vector}")
            break
    logger.plotSubgradientProcess("z_lr", "z_ip")