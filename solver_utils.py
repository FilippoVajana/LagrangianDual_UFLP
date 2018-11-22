import numpy as np
from benchmark import BenchmarkResult_v2
import model as model
import subgradient as subgradient
import pulp as pulp
import time as time

def solve(problem, solver=pulp.PULP_CBC_CMD(), console_out=False):
    solver.msg = console_out
    solver.actualSolve(problem)
    return problem

def solve_primal(instance, benchmark):    
        # print("####################")
        # print("SOLVE PRIMAL")
        # build model      
        ip_problem = model.build_primal(
                instance.c_num,
                instance.l_num,
                instance.C,
                instance.F)

        # solve
        t_s = time.time()
        solve(ip_problem, console_out=False)
        t_e = time.time()

        # save results
        benchmark.integer['x'] = np.array(get_variable_value(ip_problem, 'x').reshape(instance.c_num, instance.l_num))
        benchmark.integer['y'] = np.array(get_variable_value(ip_problem, 'y'))
        benchmark.integer['z'] = get_objectiveFunction_value(ip_problem)
        benchmark.integer_time = round((t_e - t_s), 3)

        # print(f"X = \n {benchmark.integer['x']}")
        # print(f"Y = {benchmark.integer['y']}")
        # print(f"Z = {benchmark.integer['z']}")
        # print(f"Time = {benchmark.integer_time}")
        return benchmark

def solve_relaxed(instance, benchmark):
        # print("####################")
        # print("SOLVE INTEGER RELAXED")
        # build model      
        relaxed_problem = model.build_relaxed(
                instance.c_num,
                instance.l_num,
                instance.C,
                instance.F)
        
        # solve
        t_s = time.time()
        solve(relaxed_problem, console_out=False)
        t_e = time.time()

        # save results
        benchmark.relaxed['x'] = np.array(get_variable_value(relaxed_problem, 'x').reshape(instance.c_num, instance.l_num))
        benchmark.relaxed['y'] = np.array(get_variable_value(relaxed_problem, 'y'))
        benchmark.relaxed['z'] = get_objectiveFunction_value(relaxed_problem)
        benchmark.relaxed_time = round((t_e - t_s), 3)

        # print(f"X = \n {benchmark.relaxed['x']}")
        # print(f"Y = {benchmark.relaxed['y']}")
        # print(f"Z = {benchmark.relaxed['z']}")
        # print(f"Time = {benchmark.relaxed_time}")

        return benchmark

def solve_lagrangian(instance, benchmark):    
        # print("####################")
        # print("SOLVE LAGRANGE DUAL") 
        # run subgradient algorithm
        benchmark = subgradient.subgradient(instance, benchmark) 
        
        # print(f"X = \n {benchmark.lagrange['x']}")
        # print(f"Y = {benchmark.lagrange['y']}")
        # print(f"Z = {benchmark.lagrange['z']}")
        # print(f"Time = {benchmark.lagrange_time}")
        
        return benchmark



# Auxiliary functions
def test_pulp():
    pulp.pulpTestAll()

def print_all_variables(problem):
    for v in problem.variables():
        print(v.name, "=", v.varValue)

def get_objectiveFunction_value(problem):
    return pulp.value(problem.objective)

def get_variable_value(problem, name):
    l = []
    for v in problem.variables():
        v_name = str(v.name).split('_')
        if v_name[0].lower() == name.lower():
            l.append(v.varValue)
            
    return np.array(l)
