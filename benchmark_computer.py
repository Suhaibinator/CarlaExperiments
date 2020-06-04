
from par_gcp_phys_barebone import Game # This will be replaced by the CARLA-Tasks module
from neural_net_torch import Skill_only_Net as S_o_net # Skill-only Model

import os
import math
import copy
import array
import random
import pickle
import time
import numpy as np
from scoop import futures
from subprocess import call
import torch

from deap import base, creator, tools

SEED = 2
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

f = open('ScalerData', 'rb')
scaler = pickle.load(f)
f.close()

#-----------------------------------------------------------------------------
# Hyperparameters
NUM_WORKERS = 10

n_obs = 5 # number of inputs
n_actions = 2 # number of outputs
net_sample = S_o_net(n_obs, n_actions)
NDIM = net_sample.computeTotalNumberOfParameters()

# These parameters will be replaced with the ones defined in CARLA
base_phys = {'flwf': 3.5, 'frwf': 3.5, 'rlwf': 3.5, 'rrwf': 3.5, 'mass': 2090,
             'flwmsa': 70, 'frwmsa': 70, 'speed': 60, 'steer1': 0.9,
             'steer2': 0.8, 'steer3': 0.7, 'torque1': 500.76} 
percent = 0.2 # +/- 20% variation of the nominal task parameters
REPEATS = 5
dimensions = [('steer1', 'steer2', 'steer3'), ('torque1',)]

#-----------------------------------------------------------------------------
# Helper functions

def genotype_to_phenotype(vector):
    # net_sample is needed to build the net from the vector
    # Individuals in the populations are represented as lists. 
    # In order to evaluate their performance, they need to be converted to 
    # neural network (phenotype)
    net_copy = copy.deepcopy(net_sample)
    vector_copy = copy.deepcopy(np.array(vector))
    
    for p in net_copy.parameters():
        len_slice = p.numel()
        replace_with = vector_copy[0:len_slice].reshape(p.data.size())
        p.data = torch.from_numpy( replace_with )
        vector_copy = np.delete(vector_copy, np.arange(len_slice))    

    return net_copy

def gen_tasks():
    tasks = list()
    for dimension in dimensions:
        for i in range(REPEATS):
            current_phys = base_phys.copy()
            multiplier = random.uniform(1-percent, 1+percent)
            for parameter in dimension:
                current_phys[parameter]*=multiplier
            tasks.append(current_phys)
    return tasks
    
def eval_jobs(job):
    result = []
    for i in job[1]:
        # i[0] is the index of the individual in invalid_ind
        # i[1] is the actual reference to that individual
        # i[2] is the physics parameters on which to evaluate
        f0, f1 = Game(genotype_to_phenotype(i[1]), scaler, job[0], i[2])
        result.append((i[0], f0, f1))
    return result

#-----------------------------------------------------------------------------

creator.create("FitnessMaxMin", base.Fitness, weights=(-1.0, -1.0)) # Min: f1 & Min: f2
creator.create("Individual", array.array, typecode='f', fitness=creator.FitnessMaxMin, params=None)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.randn)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, NDIM)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", eval_jobs)
toolbox.register("select", tools.selNSGA2)
toolbox.register("map", futures.map)

def main():
    gen = 0
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "min", "max"
    
    
    with open("gen237_CS1_checkpoint.pkl","rb") as file:
        cp = pickle.load(file)
    pop = cp["population"]
    logbook = cp["logbook"]
    gen = cp["generation"]
    random.setstate(cp["rndstate"])
    
    
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop] # Genotype
    
    
    num_per_worker = [math.floor(len(invalid_ind*REPEATS*len(dimensions))/NUM_WORKERS) for i in range(NUM_WORKERS)]
    for i in range(len(invalid_ind*REPEATS*len(dimensions))%NUM_WORKERS):
        num_per_worker[i]+=1
    
    jobs = []
    for i in range(len(invalid_ind)):
        for task in gen_tasks():
            jobs.append((i, task))
    
    last_one = 0
    worker_jobs = []
    for i in range(len(num_per_worker)):
        worker_jobs.append((2000+2*i, [(job[0], invalid_ind[job[0]], job[1]) for job in jobs[last_one:last_one+num_per_worker[i]]]))
        last_one+=num_per_worker[i]
    
    results = toolbox.map(toolbox.evaluate, worker_jobs)
    fitnesses = [[0, 0] for ind in invalid_ind]
    for result in results:
        for job in result:
            fitnesses[job[0]][0]+=job[1]
            fitnesses[job[0]][1]+=job[2]
    
    for i in range(len(invalid_ind)):
        ind = invalid_ind[i]
        ind.fitness.values = (fitnesses[i][0]/(REPEATS*len(dimensions)), fitnesses[i][1]/(REPEATS*len(dimensions)))
    
    return pop

if __name__ == "__main__":
    start_time = time.time()
    pop = main()
    end_time = time.time()
    print("Evaluated " + str(len(pop)) + " individuals, and " + str(len(gen_tasks())) + " tasks, using " + str(NUM_WORKERS) + " workers, in " + str(end_time-start_time) + "seconds.")
    

