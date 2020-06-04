
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
NGEN = 700
MU = 60
CXPB = 0.9
REPEATS = 5
BOUND_LOW, BOUND_UP = -5.0, 5.0

n_obs = 5 # number of inputs
n_actions = 2 # number of outputs
net_sample = S_o_net(n_obs, n_actions)
NDIM = net_sample.computeTotalNumberOfParameters()

# These parameters will be replaced with the ones defined in CARLA
base_lift, base_gravity, base_fwd_flap, base_drag = -12., 1., 5., 1. 
percent = 0.2 # +/- 20% variation of the nominal task parameters 


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


def eval_fitness(ind, scaler, port):
    f0 = 0
    f1 = 0
    net = genotype_to_phenotype(ind)
    base_phys = {'flw': 0.35, 'frw': 0.35, 'rlw': 0.35, 'rrw': 0.35, 'mass': 10000}
    current_phys = base_phys.copy()
    for i in range(5):
        current_phys['flw'] = random.uniform(0.9*base_phys['flw'], 1.1*base_phys['flw'])
        current_phys['frw'] = random.uniform(0.9*base_phys['frw'], 1.1*base_phys['frw'])
        temp_f0, temp_f1 = Game(net, scaler, port, current_phys)
        f0 += temp_f0
        f1 += temp_f1
    current_phys = base_phys.copy()
    for i in range(5):
        current_phys['flw'] = random.uniform(0.9*base_phys['flw'], 1.1*base_phys['flw'])
        current_phys['frw'] = random.uniform(0.9*base_phys['frw'], 1.1*base_phys['frw'])
        current_phys['rlw'] = random.uniform(0.9*base_phys['rlw'], 1.1*base_phys['rlw'])
        current_phys['rrw'] = random.uniform(0.9*base_phys['rrw'], 1.1*base_phys['rrw'])
        temp_f0, temp_f1 = Game(net, scaler, port, current_phys)
        f0 += temp_f0
        f1 += temp_f1
    current_phys = base_phys.copy()
    for i in range(5):
        current_phys['mass'] = random.uniform(0.9*base_phys['mass'], 1.1*base_phys['mass'])
        
        f0 += temp_f0
        f1 += temp_f1
    f0/=15
    f1/=15
    return f0, f1
    
def eval_jobs(job):
    result = []
    for i in job[1]:
        #f0, f1 = eval_fitness(i, scaler, job[0])
        f0, f1 = Game(genotype_to_phenotype(i), scaler, job[0], None)
        result.append((f0, f1))
    return result

#-----------------------------------------------------------------------------

creator.create("FitnessMaxMin", base.Fitness, weights=(-1.0, -1.0)) # Min: f1 & Min: f2
creator.create("Individual", array.array, typecode='f', fitness=creator.FitnessMaxMin, params=None)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.randn)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, NDIM)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", eval_jobs)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)
toolbox.register("map", futures.map)

def main():
    gen = 0
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "min", "max"
    
    pop = toolbox.population(n=MU)
    """
    with open("gen502_CS1_checkpoint.pkl","rb") as file:
        cp = pickle.load(file)
    pop = cp["population"]
    logbook = cp["logbook"]
    gen = cp["generation"]
    random.setstate(cp["rndstate"])
    """
    
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid] # Genotype
    
    num_per_worker = [math.floor(len(invalid_ind)/NUM_WORKERS) for i in range(NUM_WORKERS)]
    for i in range(len(invalid_ind)%NUM_WORKERS):
        num_per_worker[i]+=1
    
    last_one = 0
    jobs = []
    for i in range(len(num_per_worker)):
        jobs.append((2000+2*i, invalid_ind[last_one:last_one+num_per_worker[i]]))
        last_one+=num_per_worker[i]
    
    results = toolbox.map(toolbox.evaluate, jobs)
    fitnesses = []
    for result in results:
        fitnesses+=result
    
    
    # for ind in invalid_ind:
    #     fitness = toolbox.evaluate(ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    
    record = stats.compile(pop)
    logbook.record(gen=gen, evals=len(invalid_ind), **record)
    print(logbook.stream)

    cp = dict(population=pop, generation=gen, logbook=logbook, rndstate=random.getstate())
    with open("gen%d_CS1_checkpoint.pkl" % (gen), "wb") as cp_file:
        pickle.dump(cp, cp_file)

    met_criterion = False
    # Begin the generational process
    for gen in range(gen+1, NGEN+1):
        if not met_criterion or True:#not stop_run:
            # Vary the population
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]
            
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= CXPB:
                    toolbox.mate(ind1, ind2)
                
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values
        
            # Evaluate the individuals with an invalid fitness (with the same seed no.)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid] # Genotype
    
            num_per_worker = [math.floor(len(invalid_ind)/NUM_WORKERS) for i in range(NUM_WORKERS)]
            for i in range(len(invalid_ind)%NUM_WORKERS):
                num_per_worker[i]+=1
    
            last_one = 0
            jobs = []
            for i in range(len(num_per_worker)):
                jobs.append((2000+2*i, invalid_ind[last_one:last_one+num_per_worker[i]]))
                last_one+=num_per_worker[i]
    
            results = toolbox.map(toolbox.evaluate, jobs)
            fitnesses = []
            for result in results:
                fitnesses+=result
    
    
            # for ind in invalid_ind:
            #     fitness = toolbox.evaluate(ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population
            pop = toolbox.select(pop + offspring, MU)
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            print(logbook.stream)

            cp = dict(population=pop, generation=gen, logbook=logbook, rndstate=random.getstate())
            with open("gen%d_CS1_checkpoint.pkl" % (gen), "wb") as cp_file:
                pickle.dump(cp, cp_file)
                
            if gen > 0:
                call("rm gen%d_*" % (gen-1), shell=True)

            met_fitness = 0
            for ind in pop:
                if ind.fitness.values[1] <= 20:
                    met_fitness+=1
            met_criterion = met_fitness >= 10

            if (gen == NGEN):
                pop_as_nets = []
                for ind in pop:
                    net = genotype_to_phenotype(ind)
                    net.fitness = ind.fitness.values
                    pop_as_nets.append( net )

    # print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))
    #print("Final population hypervolume is %f" % hypervolume(pop))
    
    return pop_as_nets

if __name__ == "__main__":

    pop = main()
    pickle.dump(pop, open("popAsNets.p","wb"))

