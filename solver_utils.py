import numpy as np
from pulp import *


def test_pulp():
    pulpTestAll()


def solve(problem, solver=PULP_CBC_CMD(), console_out=False):
    solver.msg = console_out
    solver.actualSolve(problem)

    return problem

def print_all_variables(problem):
    for v in problem.variables():
        print(v.name, "=", v.varValue)

def get_objectiveFunction_value(problem):
    return value(problem.objective)

def get_variable_value(problem, name):
    l = []

    for v in problem.variables():
        v_name = str(v.name).split('_')
        if v_name[0].lower() == name.lower():
            l.append(v.varValue)
            
    return np.array(l)
