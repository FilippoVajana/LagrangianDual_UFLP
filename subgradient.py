import numpy as np
from benchmark import BenchmarkResult_v2
import model as model
import solver_utils as solver
import time as time

def subgradient(instance, benchmark):
    '''
    Solves Lagrangian Dual problem using subgradient algorithm
    '''   
    # init data
    c_num = instance.c_num #clients number
    l_num = instance.l_num #locations number
    u_vector = instance.U #initial lagrange multipliers
    z_ip = benchmark.integer['z'] #best primal
    C = instance.C #profits matrix
    F = instance.F #costs vector

    # init params
    t_max = c_num #loops w/ improvement
    k = 1 #no improvement loop counter
    alpha_stop = 0.005 #stop
    alpha = 1
    z_best = np.Infinity
    problem_best = None

    # algorithm body
    t_s = time.time()
    while True:
        # compute best z(u)
        problem = model.build_lagrange(c_num, l_num, u_vector, C, F)
        solver.solve(problem)
        z_lr = solver.get_objectiveFunction_value(problem)
        benchmark.sg_z.append(z_lr) #log  

        # compute subgradient
        x_lr = solver.get_variable_value(problem, 'x')        
        subgradient = subgradient_tbn(x_lr, c_num, l_num)
        
        # check subgradient
        if not np.any(subgradient):
            # print(f"null subgradient:\n {subgradient}")
            problem_best = problem
            break

        # compute step size            
        step = step_tbn(alpha, z_ip, z_lr, subgradient)
        benchmark.sg_step.append(step) #log
        
        # compute new lagrange multipliers
        u_vector = update_lagrange_tbn(u_vector, step, subgradient)

        # check bound
        if round(z_lr, 2) < round(z_best, 2):
            z_best = z_lr
            problem_best = problem
            k = 0
            #print(f"z_best: {z_best}")
            
        # check counter
        if k == t_max:
            alpha = alpha / 2
            k = 0
            #print(f"alpha: {alpha}")

        # check stop condition
        if alpha < alpha_stop:
            break
        
        k = k + 1

    t_e = time.time()   

    # save results
    benchmark.lagrange['x'] = np.array(solver.get_variable_value(problem_best, 'x').reshape(c_num, l_num))
    benchmark.lagrange['y'] = np.array(solver.get_variable_value(problem_best, 'y'))
    benchmark.lagrange['z'] = solver.get_objectiveFunction_value(problem_best).item()
    benchmark.lagrange_time = round((t_e - t_s), 2)
    
    return benchmark


def subgradient_tbn(x_lr, c_num, l_num):
    '''
    Computes subgradient vector
    '''
    # reshape X matrix
    x_lr = np.reshape(x_lr, (c_num, l_num)) 
    # compute D*x(u)
    d_x = [sum([x_lr[i][j] for j in range(l_num)]) for i in range(c_num)]
    # d vector
    d = np.ones(c_num)
    # subgradient
    subgradient = d - d_x
    return subgradient

def step_tbn(alpha, z_ip, z_lr, subgradient):
    '''
    Computes step size
    '''
    num = np.abs(z_ip - z_lr)
    den = (np.linalg.norm(subgradient)) ** 2
    step = alpha * num / den
    return step

def update_lagrange_tbn(u, step, subgradient):
    '''
    Computes next lagrange multipliers vector
    '''
    l = range(len(u))
    u_next = [np.max(u[i] - step * subgradient, 0) for i in l]
    #u_next = [u[i] - step * subgradient for i in l]
    return u_next