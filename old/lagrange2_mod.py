from pulp import *
from solver_utils import *
from data_utils import *
import numpy as np

class UFLWolsey():
    C = None
    F = None
    clients = 0
    locations = 0
    dg = DataGenerator()
    logger = None

    def __init__(self, clients, locations, c_matrix, f_vector, logger):
        self.clients = clients
        self.locations = locations
        self.C = c_matrix
        self.F = f_vector
        self.logger = logger

    def BuildIpProblem(self):
        # problem instance
        problem = LpProblem("UFL - Primal Problem (Wolsey)", sense=LpMaximize)

        # decision variables
        x = self.dg.VAR_matrix(self.clients, self.locations, 'x', 'Binary')
        y = self.dg.VAR_vector(self.locations, 'y', 'Binary')

        # objective function
        service_costs = [self.C[i][j] * x[i][j] for i in range(self.clients) for j in range(self.locations)]
        fixed_costs = [self.F[j] * y[j] for j in range(self.locations)]
        problem += lpSum(service_costs) - lpSum(fixed_costs), "Total Costs"

        # constraints
        for i in range(self.clients):
            problem += lpSum([x[i][j] for j in range(self.locations)]) == 1
        
        for i in range(self.clients):
            for j in range(self.locations):
                problem += x[i][j] - y[j] <= 0

        return problem
    def BuildLrProblem(self, u_vector):
        """
        Lagrange relaxation for UFL problem as in Wolsey
        """

        # problem instance
        problem = LpProblem("UFL - Lagrange Relaxed Problem (Wolsey)", sense=LpMaximize)

        # check lagrange multipliers vector
        if len(u_vector) != self.clients : 
            raise ValueError('Invalid u_vect size')

        # decision variables    
        x = dg.VAR_matrix(self.clients, self.locations, 'x', 'Binary')
        y = dg.VAR_vector(self.locations, 'y', 'Binary')

        # objective function
        s_c = [(self.C[i][j] - u_vector[i]) * x[i][j] for i in range(self.clients) for j in range(self.locations)]
        f_c = [self.F[j] * y[j] for j in range(self.locations)]
        u_c = [u_vector[i] for i in range(self.clients)]

        problem += lpSum(s_c) - lpSum(f_c) + lpSum(u_c)

        # constraints
        for i in range(self.clients):
            for j in range(self.locations):
                problem += x[i][j] - y[j] <= 0

        return problem

#Subgradient    
    def subgradient_tbn(self, x_lr):
        # reshape X matrix
        x_lr = np.reshape(x_lr, (self.clients, self.locations)) 

        # compute D*x(u)
        d_x = [sum([x_lr[i][j] for j in range(self.locations)]) for i in range(self.clients)]
        # print(x_lr)
        # print(d_x)

        # d vector
        d = np.ones(self.clients)

        # subgradient
        sg = d - d_x
        #print(f"subgradient: \n{sg}")

        return sg

#Step size    
    def step_tbn(self, alpha, z_ip, z_lr, subgradient):
        num = np.abs(z_ip - z_lr)
        den = (np.linalg.norm(subgradient)) ** 2
        step = alpha * num / den

        self.logger.logData("norm", den)
        #print(f"step_tbn: {step}")
        return step

#Update multipliers    
    def update_lagrange_tbn(self, u, step, subgradient):
        u_next = u + step * subgradient

        #print(f"u_next_tbn: \n{u_next}")
        return u_next
        
#Run subgradient
    def solve_subgradient_tbn(self, u_vector, z_ip):
        #parameters
        t_max = 10 #maximum loops w/ improvement
        k = 1 #no improvement loop counter
        alpha_stop = 0.05 #stop parameter
        alpha = 10
        z_best = np.Infinity 

        while True:
            #compute best z(u)
            p_lr = self.BuildLrProblem(u_vector)
            solve(p_lr)
            z_lr = get_objectiveFunction_value(p_lr)
            self.logger.logData('z_lr', z_lr)
            self.logger.logData('z_ip', z_ip) #simplify plotting

            #compute subgradient
            x_lr = get_variable_value(p_lr, 'x')
            sg = self.subgradient_tbn(x_lr)

            #check subgradient
            if not np.any(sg):
                break

            #compute step size            
            step = self.step_tbn(alpha, z_ip, z_lr, sg)
            self.logger.logData('step_size', step)

            #compute new lagrange multipliers
            u = u_vector #copy
            u_vector = self.update_lagrange_tbn(u, step, sg)

            #check bound
            if z_lr < z_best:
                z_best = z_lr
                k = 0
                print(f"z_best: {z_best}")
            else:
                k = k+1

            #check counter
            if k == t_max:
                alpha = alpha / 2
                k = 0
                print(f"alpha: {alpha}")

            #check stop condition
            if alpha < alpha_stop:
                break
        
        return z_best

if __name__ == '__main__':
    clients = 50
    locations = 3
    
    dg = DataGenerator()
    # init clients revenues
    c_matrix = dg.INT_matrix(clients, locations)

    # init facilities service cost
    f_avg_profit = (np.average(c_matrix) * clients) / locations # avg facility profit   
    f_min_cost = f_avg_profit * 0.25
    f_max_cost = f_avg_profit * 2
    f_vector = dg.INT_vector(locations, f_min_cost, f_max_cost, False)
    
    # lagrange multipliers
    # u_vector = dg.INT_vector(clients)
    # u_vector = np.zeros(clients)
    u_vector = dg.INT_vector(clients, f_min_cost, f_max_cost)

    # problem class instance
    logger = DataLogger()
    ufl = UFLWolsey(clients, locations, c_matrix, f_vector, logger)

    # IP problem      
    ip_problem = ufl.BuildIpProblem()
    print("Solve primal problem")
    solve(ip_problem, console_out=False)
    print(f"primal Y: {get_variable_value(ip_problem, 'y')}")
    print(f"primal value: {get_objectiveFunction_value(ip_problem)}")

    # LR problem (tbn)
    z_ip = get_objectiveFunction_value(ip_problem)
    lr_problem = ufl.solve_subgradient_tbn(u_vector, z_ip)
    print(lr_problem)

    #plot data
    import matplotlib.pyplot as plt

    #subgradient bound improvement
    plt.figure(1)
    #plt.subplot(411)
    z_subgradient = logger.d['z_lr']
    z_ip = logger.d["z_ip"]
    plt.plot(z_subgradient, linestyle='dashed')
    plt.plot(z_ip, color='orange')
    plt.xlabel('Loops')
    plt.ylabel('Bound Value')
    plt.title('Subgradient Bound')
    plt.grid(True)
    #plt.show()

    #subgradient step size
    plt.figure(2)
    #plt.subplot(412)
    step_size = logger.d["step_size"]
    plt.plot(step_size)
    plt.xlabel('Loops')
    plt.ylabel('Step Size')
    plt.title('Subgradient Step Size')
    plt.grid(True)

    plt.show()
