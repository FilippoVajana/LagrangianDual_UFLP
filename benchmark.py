import numpy as np
import pandas as pd
import random as rnd
import jsonpickle as json
import os
import time
import datetime
import logging as log
log.basicConfig(level=log.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')

# POJO classes
class InstanceData():
    """
    Represents a problem's instance data
    """
    min = 0
    max = 10
        
    def create(self, clients, locations):
        self.c_num = clients
        self.l_num = locations
        self.id = '{row}x{col}_{hash}'.format(row=self.c_num, col=self.l_num, hash=str(rnd.random())[2:6])
        # generate data        
        self.U = np.zeros(self.c_num, dtype=int)
        self.C = np.random.randint(low=self.min, high=self.max+1, size=(self.c_num, self.l_num))        
        avg_cp = [np.sum(self.C[:,i]) for i in range(self.l_num)]
        self.F = [np.random.randint(low=avg_cp[i]*0.5, high=avg_cp[i]*1.2+1) for i in range(self.l_num)]
        return self

    def encode(self):
        # flatten and byte encode numpy array
        self.U = self.U.reshape(self.c_num)
        self.U = self.U.tostring()
        #print(self.U)

        self.C = np.asarray(self.C)
        self.C = self.C.reshape(self.c_num * self.l_num)
        self.C = self.C.tostring()
        #print(self.C)

        self.F = np.asarray(self.F).reshape(self.l_num)
        self.F = self.F.tostring()
        #print(self.F)
        return self
    
    def decode(self):
        self.U = np.frombuffer(self.U, dtype=int)
        #print(self.U)

        self.C = np.frombuffer(self.C, dtype=int).reshape((self.c_num, self.l_num))
        #print(self.C)        

        self.F = np.frombuffer(self.F, dtype=int)
        #print(self.F)        
        return self

    def print(self):
        print("U = ", self.U)        
        print("F = ", self.F)
        print("C = ", self.C)
class Benchmark():
    DATA_PATH = None
    def __init__(self):
        # set data folder
        path = os.path.join(os.getcwd(), "data")    
        if not (os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        self.DATA_PATH = path

    def save_instances(self, instances):
        for i in instances:
            # serialize instance
            data = json.encode(i.encode())
            # create file
            name = "instance_{}.json".format(i.id)
            f_path = os.path.join(self.DATA_PATH, name)
            file = open(f_path, "w+")
            file.write(data)
            file.close()

    def load_instances(self):
        # get files in data folder
        paths = [os.path.join(self.DATA_PATH, f) for f in os.listdir(self.DATA_PATH) if os.path.isfile(os.path.join(self.DATA_PATH, f))] 
        # load instances
        instances = []
        for p in paths:
            file = open(p, 'r')
            data = file.read()
            file.close()
            i = json.decode(data, classes=InstanceData).decode()
            instances.append(i)        
        return instances

    def build_test_instances(self, shapes=[], count=1):
        instances = []
        for (c,l) in shapes:
            for _ in range(count):
                time.sleep(.5)
                instance = InstanceData().create(c,l)
                instances.append(instance)
        return instances

    def to_dataframe(self, bnc2_list):
        '''
        Converts a list of benchmark results into a pandas DataFrame
        '''
        # unfold results
        unf = [r.unfold() for r in bnc2_list]

        # build data frame
        cols = list(unf[0].keys())
        df = pd.DataFrame(data=unf, columns=cols)

        return df

    def merge_dataframe(self, dfs=[]):
        '''
        Concats test dataframes
        '''
        return pd.concat(dfs)

    def save_dataframe(self, df):
        '''
        Saves to a xlsx file the resulting benchmarks dataframe
        '''
        dt = datetime.datetime.now().strftime("%d%m_%H%M")
        name = "benchmark_{}.xlsx".format(dt)
        path = os.path.join(self.DATA_PATH, name)
        df.to_excel(path, index=False)
    
    def load_dataframe(self, name):
        '''
        Loads benchmark dataframe
        '''
        path = os.path.join(self.DATA_PATH, name)
        df = pd.read_excel(path)
        return df

class InstanceData_v2():
    """
    Instance data
    """
    def create(self, clients, locations):
        self.c_num = clients
        self.l_num = locations        
        self.id = '{row}x{col}_{hash}'.format(row=self.c_num, col=self.l_num, hash=str(rnd.random())[2:6])

        # generate data        
        self.U = np.zeros(self.c_num, dtype=int)
        self.C = np.random.randint(low=0, high=10+1, size=(self.c_num, self.l_num))        
        avg_cp = [np.sum(self.C[:,i]) for i in range(self.l_num)]
        self.F = [np.random.randint(low=avg_cp[i]*0.5, high=avg_cp[i]*1.2+1) for i in range(self.l_num)]
        return self

    def encode(self):
        self.U = np.array(self.U).flatten().tostring()
        self.C = np.array(self.C).flatten().tostring()
        self.F = np.array(self.F).flatten().tostring()
        return self

    def decode(self):
        self.U = np.frombuffer(self.U, dtype=int)
        self.C = np.frombuffer(self.C, dtype=int).reshape((self.c_num, self.l_num))
        self.F = np.frombuffer(self.F, dtype=int)
        return self

    def print(self):
        print("U = ", self.U)        
        print("F = ", self.F)
        print("C = ", self.C)

class BenchmarkResult_v2():    
    '''
    Benchmark result
    '''
    def __init__(self, instance):
        # instance params
        self.id = instance.id
        self.clients = instance.c_num
        self.locations = instance.l_num  
        # solver params    
        self.integer = {'x':[],'y':[],'z':0}
        self.relaxed = {'x':[],'y':[],'z':0}
        self.lagrange = {'x':[],'y':[],'z':0}
        # subgradient params
        self.sg_z = []
        self.sg_step = []
        # performance
        self.integer_time = 0.0
        self.relaxed_time = 0.0
        self.lagrange_time = 0.0

    def unfold_dict(self,d):   
        '''
        Input: a dictionary
        Output: unfolded dictionary items
        '''
        res = {}    
        for k,v in d.items():
            #print(k,':',v)
            if not isinstance(v,dict):
                res[k] = v
            else:      
                #fix and add sub entries
                for s_k,s_v in self.unfold_dict(v).items(): 
                    res[k+'_'+s_k] = s_v             
        return res
    def unfold(self):
        '''
        Unfold a BenchmarkResult_v2 instance
        '''
        # get attributes
        var = {k:v for (k,v) in vars(self).items() if not k.startswith('__') and not callable(k)}

        # unfold result
        unf = self.unfold_dict(var)

        #HACK: check type, remove after
        # for k,v in unf.items():
        #     print(k, type(v))
        return unf

class Runner:
    '''
    Benchmark runner
    '''
    def build_instances(self, shapes, count):
        log.info(f'Build {len(shapes) * count} instances')
        instances = []
        for (c,l) in shapes:
            for _ in range(count):
                time.sleep(.5)
                instance = InstanceData_v2().create(c,l)
                instances.append(instance)
        return instances

    def init_folders(self):
        log.info('Initialize folders')
        # create main data folder
        path = os.path.join(os.getcwd(), "data")
        if not (os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        # create run subfolder
        name = datetime.datetime.now().strftime("%d%m_%H%M")
        name = datetime.datetime.now().strftime("%d%m") #remove after testing
        run_path = os.path.join(path, name)
        if not os.path.exists(run_path):
            os.mkdir(run_path)
        # save paths        
        self.DATA_PATH = path
        self.RUN_PATH = run_path

    def save_instances(self, instances):
        # init data foldes
        self.init_folders()
        # flatten data
        log.info('Flatten instances')
        for i in instances:            
            i = i.encode() 
        # serialize instances
        log.info('Serialize instances')
        data = json.encode(instances)
        # create json file
        path = os.path.join(self.RUN_PATH, "data.json")
        file = open(path, "w+")            
        # write json file
        log.info(f'Save instances to {path}')
        file.write(data)
        file.close()

    def save_dataframe(self, df):
        '''
        Saves to a xlsx file the resulting benchmarks dataframe
        '''                
        name = "result.xlsx"
        path = os.path.join(self.RUN_PATH, name)
        log.info(f'Save dataframe to {path}')
        df.to_excel(path, index=False)

    def load_instances(self, path):
        # check path
        if not os.path.exists(path):
            log.error(f'Invalid path {path}')
            raise Exception("Invalid Path")

        # get json data file  
        log.info(f'Load instances from {path}')    
        file_path = [os.path.join(path,p) for p in os.listdir(path) if "data.json" in p][0]
        file = open(file_path, 'r')
        data = file.read()
        file.close()

        # deserialize data
        log.info(f'Deserialize instances')
        data = [i.decode() for i in json.decode(data, classes=InstanceData_v2)]
        return data
    
    def load_dataframe(self, path):
        # check path
        if not os.path.exists(path):
            log.error(f'Invalid path {path}')
            raise Exception("Invalid Path")

        # get json data file  
        log.info(f'Load dataframe from {path}')    
        file_path = [os.path.join(path,p) for p in os.listdir(path) if "result.xlsx" in p][0]
        # file = open(file_path, 'r')
        # data = file.read()
        # file.close()

        # read excel data
        log.info('Read dataframe data')
        df = pd.read_excel(file_path, index_col=0)

        return df


    def to_dataframe(self, bnc2_list):
        '''
        Converts a list of benchmark results into a pandas DataFrame
        '''
        log.info('Unfold result objects')
        # unfold results
        unf = [r.unfold() for r in bnc2_list]

        # flatten data
        log.info('Flatten instances')
        for b in unf:     
            for k,v in b.items():
                if isinstance(v, np.ndarray):
                    b[k] = v.flatten()
                if isinstance(v, list):
                    b[k] = np.array(v)

        # build data frame
        log.info('Build dataframe')
        cols = list(unf[0].keys())
        df = pd.DataFrame(data=unf, columns=cols)

        return df
    
    def run_benchmark(self, shapes=[], count=0, dir=None):
        t_start = time.time()
        # get data
        data = []
        if dir is str:            
            data = self.load_instances(dir)
        else:
            data = self.build_instances(shapes, count)
        
        # run
        import solver_utils as solver
        results = []
        for idx,i in enumerate(data):
            log.info(f'Instance #{idx}')
            res = BenchmarkResult_v2(i)
            log.info('Solve primal')
            res = solver.solve_primal(i, res)
            log.info('Solve relaxed')
            res = solver.solve_relaxed(i, res)
            log.info('Solve lagrange')
            res = solver.solve_lagrangian(i, res)
            results.append(res)

        # save data file
        self.save_instances(data)

        # save result file
        df = self.to_dataframe(results)
        self.save_dataframe(df)

        t_end = time.time()
        log.info(f'Benchmark run in: {round((t_end - t_start) / 60, 1)} min')

        return df
        


if __name__ == '__main__':
    c = [10,20,40,80,160]
    l = [2,4,8,16]
    shapes = np.array(np.meshgrid(c,l)).T.reshape(-1,2)    
    repeat = 6    
    
    # run benchmark
    runner = Runner()
    runner.run_benchmark(shapes,repeat)



