import numpy as np
import jsonpickle as json
import os

class InstanceData():
    """
    Represents a problem's instance data
    """
    min = 0
    max = 10
        
    def create(self, clients, locations):
        self.c_num = clients
        self.l_num = locations

        # generate data        
        self.U = np.zeros(self.c_num, dtype=int)
        self.C = np.random.randint(low=self.min, high=self.max+1, size=(self.c_num, self.l_num))        
        avg_cp = [np.average(self.C[i]) for i in range(self.c_num)]
        self.F = [np.random.randint(low=avg_cp[i]/2, high=avg_cp[i]+1) for i in range(self.l_num)]  #check dimension

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

        self.F = np.asarray(self.F).reshape(self.l_num) #check dimension
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
 



def build_test_instances(shapes=[], count=1):
    instances = []
    for (c,l) in shapes:
        for _ in range(count):
            instance = InstanceData().create(c,l)
            instances.append(instance)
    return instances

DATA_PATH = None
def save_instance(instance, id):
    # check data folder
    setup_data_folder()
    # serialize instance
    data = json.encode(instance.encode())
    # create file
    name = "instance_{}_{}_{}.json".format(id, instance.c_num, instance.l_num)
    f_path = os.path.join(DATA_PATH, name)
    file = open(f_path, "w+")
    # write data
    file.write(data)
    # close file
    file.close()

def load_instance(instance, id):
    # read file
    name = "instance_{}_{}_{}.json".format(id, instance.c_num, instance.l_num)
    f_path = os.path.join(DATA_PATH, name)
    file = open(f_path, "r")
    # deserialize instance
    data = file.read()
    i = json.decode(data).decode()
    file.close()
    return i

def load_all():
    # get files in data folder
    paths = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))] 
    # load instances
    instances = []
    for p in paths:
        file = open(p, 'r')
        data = file.read()
        i = json.decode(data, classes=InstanceData).decode()
        instances.append(i)
        file.close()
    return instances

def setup_data_folder():    
    path = os.path.join(os.getcwd(), "data")    
    if not (os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)
    # set data directory path
    global DATA_PATH
    DATA_PATH = path

if __name__ == '__main__':
    s = [
        (10,2),
        (20,4),
        (30,6),
        (50,10)
    ]
    c = 5
    l = build_test_instances(s, c)
    for idx, instance in enumerate(l):
        save_instance(instance, idx)    
    loaded = load_all()
    print("instances loaded: ", len(loaded))
    for i in loaded:
        i.print()
