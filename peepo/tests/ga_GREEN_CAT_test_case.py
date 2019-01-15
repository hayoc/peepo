14/01/2019
import json
import logging
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import copy

from peepo.predictive_processing.v3.genetic_algorithm import GeneticAlgorithm
from peepo.predictive_processing.v3.peepo_network import PeepoNetwork

from peepo.predictive_processing.v3.generative_model import GenerativeModel
from pomegranate.distributions.DiscreteDistribution import DiscreteDistribution
from peepo.predictive_processing.v3.sensory_input import SensoryInput
from peepo.predictive_processing.v3.utils import get_index_matrix

LEFT = 'LEFT'
RIGHT = 'RIGHT'
UP = 'UP'
DOWN = 'DOWN'

VISION = 'VISION'
MOTOR = 'MOTOR'


def get_thruth_table():
    comb = np.transpose(get_index_matrix([2, 2, 2])).astype(int)
    thruth = np.full((8,8),0)
    thruth[0][0] =  1
    thruth[0][1] =  1
    thruth[0][2] =  1
    thruth[0][3] =  1
    thruth[0][4] =  1
    thruth[0][7] =  1
    thruth[1][0] =  1
    thruth[1][1] =  1
    thruth[1][2] =  1
    thruth[1][3] =  1
    thruth[1][4] =  1
    thruth[1][7] =  1
    thruth[2][5] =  1
    thruth[3][5] =  1
    thruth[4][6] =  1
    thruth[5][6] =  1
    thruth[6][0] = 1
    thruth[6][1] = 1
    thruth[6][2] = 1
    thruth[6][3] = 1
    thruth[6][4] = 1
    thruth[6][7] = 1
    thruth[7][0] = 1
    thruth[7][1] = 1
    thruth[7][2] = 1
    thruth[7][3] = 1
    thruth[7][4] = 1
    thruth[7][7] = 1
    return thruth, comb

GCAT = 'GREEN_CAT'
GDOG = 'GREEN_DOG'
DUMMY = 'DUMMY'
GREEN = 'GREEN'
CAT = 'CAT'
DOG = 'DOG'


class Peepo:
    """
    This organism represents peepo. Each peepo takes as parameters a name, an initial position and the list of
    obstacles present in its environment."""


    def __init__(self, name, network):
        self.name = name
        self.network = network
        self.food = 0
        self.belief = {
            GCAT: False,
            GDOG: False,
            DUMMY: False,
        }
        self.view = {
            GREEN: False,
            CAT: False,
            DOG: False,
        }
        self.loop = 0
        self.thruth_table, self.combinations = get_thruth_table()
        #self.generative_model = GenerativeModel(network, SensoryInputPeepo(self))
        self.pomegranate = network.to_pomegranate()
        # print(self.pomegranate)



    def get_observation(self,index):
        return self.combinations[index].astype(int)

    def get_belief(self,index):
        return self.combinations[index].astype(int)

    def update(self):
        # logging.info(self.name + ' : ' + str(self.x) + ' - ' + str(self.y)+ '  food gathered : ' +  str(self.food))
        error = 0
        for state_belief in range(8):
            belief = self.get_belief(state_belief).astype(int)
            belief = np.append(belief, None)
            belief = np.append(belief, None)
            belief = np.append(belief, None)
            prediction = self.pomegranate.predict([belief])
            prediction = np.asarray(prediction)
            expected = []
            [expected.append(int(x)) for i, x in enumerate(prediction[0]) if i > 2 ]
            for state_observation in range(8):
                observation = self.get_observation(state_observation)
                if np.array_equal(expected,observation):
                    error += self.thruth_table[state_belief][state_observation]
        self.food = error

def get_optimal_network(population):
    pop = []
    for x in population:
        n_edges = len(x[1].edges)
        pop.append([x[0],x[1], n_edges])
    pop = sorted(pop, key=lambda chromo: chromo[0], reverse=True)
    best_population = []
    best_score = pop[0][0]
    for x in pop:
        if x[0] != best_score:
            break
        best_population.append(x)
    best_population = sorted(best_population, key=lambda chromo: chromo[2])
    return best_population[0]





def minimum_normalized_fitness_score(average_fitness,popul):
    # populations = sorted(population, key=lambda chromo: chromo[0], reverse=True)
    non_zero_pop = []
    [non_zero_pop.append(x[0]) for x in popul if x[0] >= average_fitness]
    return np.mean(non_zero_pop)


def create_population(generation, individuals):
    pop = []
    for i, idv in enumerate(individuals):
        peepo = Peepo(name='peepo_' + str(generation) + '_' + str(i),
                      network=idv[1])
        pop.append(peepo)

    return pop


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    num_individuals = 70
    num_generations = 20
    ga = GeneticAlgorithm('green_cat', min_fitness_score = 0.0, p_mut_top = 0.2, p_mut_cpd = 0.2,Npop=num_individuals, max_removal=2)
    population = ga.get_population()
    treshold = 0
    avg_fitnesses = []
    final_network = []
    for gen in range(num_generations):
        print('*********************                     GENERATION ', gen,
                     ' *****************************************')
        peepos = create_population(gen, population)
        final_population = []
        for ind, peepo in enumerate(peepos):
            peepo.update()
            population[ind][0] = peepo.food
            final_population.append([peepo.food, population[ind][1]])
        avg_fitness, population = ga.evolve(population, treshold)
        ''' PROPOSAL FOR NORMALIZE FITNESS FOR THIS CASE          '''
        treshold = minimum_normalized_fitness_score(avg_fitness, final_population)
        if  avg_fitness < 0:
            # logging.info(' population collapsed :-( ')
            print(' population collapsed :-( ')
            break
        # logging.info('Average fitness: %d', avg_fitness)
        print('Average fitness: ', avg_fitness)
        avg_fitnesses.append(avg_fitness)
    final_network = get_optimal_network(final_population)
    print('\n\nFINAL NETWORK')
    print('________________\n\n')
    print(final_network[1].edges)
    '''TO DO perhaps : 
    browse through the final networks and make predictions and compare with expected to get the best and simpliest network?'''

    t = np.arange(0.0, len(avg_fitnesses), 1)
    fig, ax = plt.subplots()
    ax.plot(t, avg_fitnesses)
    ax.set(xlabel='generation', ylabel='average fitness',
           title='test of green cat with genetic algorithm')
    ax.grid()
    plt.show()