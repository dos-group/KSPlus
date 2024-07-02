import pandas as pd
import matplotlib.pyplot as plt
import random
import predictor
import os

class TaskInstance:
    def __init__(self, name, insize, innum, mem, cpu, max_mem):
        self.name = name
        self.insize = insize
        self.innum = innum
        self.mem = mem
        self.cpu = cpu
        self.max_mem = max_mem

    def plot(self, *args, **kwargs):
        plt.scatter(self.mem, *args, **kwargs)

class Task:
    name = ""
    instances = []

    def __init__(self, name = "", folder = None):
        self.name = name
        if folder != None:
            #taskis = []
            #for file in os.listdir(f'{folder}'):
            #    if '_metadata.csv' in file:
            #        vals = pd.read_csv(f'{folder}/{file}', skiprows=3)
            #        name = vals["name"][1]
            #        taskis.append(TaskInstance(name, vals["_value"][1], vals["_value"][0], [i for (_,i) in pd.read_csv(f'{folder}/{file[:-13]}_memory.csv', skiprows=3)['_value'].items()], [i for (_,i) in pd.read_csv(f'{folder}/{file[:-13]}_cpu.csv', skiprows=3)['_value'].items()]))
            #self.instances = sorted(taskis, key = lambda x: len(x.mem))
            self.instances = []
            for file in sorted(os.listdir(f'{folder}')):
                if '_metadata.csv' in file:
                    vals = pd.read_csv(f'{folder}/{file}', skiprows=3)
                    name = vals["name"][1]
                    max_mem = vals.query('_field == "max_mem"')['_value'].iloc[0]
                    # TODO currently quick and dirty, we could optimize read-in time, especially for bigger inputs
                    self.instances.append(TaskInstance(name, vals["_value"][1], vals["_value"][0], [i for (_,i) in pd.read_csv(f'{folder}/{file[:-13]}_memory.csv', skiprows=3)['_value'].items()], [i for (_,i) in pd.read_csv(f'{folder}/{file[:-13]}_cpu.csv', skiprows=3)['_value'].items()],[max_mem / 1000000 for _ in pd.read_csv(f'{folder}/{file[:-13]}_memory.csv', skiprows=3)['_value'].items()]))

    def split(self, train_percent, seed = 18181):
        r = random.Random(seed)
        trainidx = set()
        while len(trainidx) < int(train_percent * len(self.instances)):
            trainidx.add(int(len(self.instances)*r.random()))

        train = Task()
        data = Task()
        train.name = self.name
        data.name = self.name
        train.instances = [self.instances[i] for i in sorted(trainidx)]
        data.instances = [self.instances[i] for i in range(len(self.instances)) if i not in trainidx]

        return (train,data)

    def get_predictor(self, num_segments = predictor.default_num_segments):
        return predictor.Predictor(self.instances, num_segments=num_segments)
