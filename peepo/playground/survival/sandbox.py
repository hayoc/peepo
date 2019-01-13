import json
import logging
import random

import matplotlib.pyplot as plt
import numpy as np
from pathos.multiprocessing import ProcessPool

from peepo.playground.survival.organism import Peepo, Food
from peepo.predictive_processing.v3.genetic_algorithm import GeneticAlgorithm

MAP_SIZE = (800, 800)


def generate_food(num, offset=20):
    objects = []
    for x in range(0, num):
        objects.append({
            'name': 'food_' + str(x),
            'x': random.randint(offset, MAP_SIZE[0] - offset),
            'y': random.randint(offset, MAP_SIZE[1] - offset)
        })
    with open('food.json', 'w') as outfile:
        json.dump(objects, outfile)


def read_food():
    food_stuff = []
    with open('food.json') as json_data:
        for f in json.load(json_data):
            food_stuff.append(Food(f['name'], (f['x'], f['y'])))
    return food_stuff


def create_population(generation, individuals, food):
    pop = []
    for i, idv in enumerate(individuals):
        peepo = Peepo(name='peepo_' + str(generation) + '_' + str(i),
                      network=idv[1],
                      pos=(5, 5),
                      obstacles=food)
        pop.append(peepo)

    return pop


def minimum_normalized_fitness_score(average_fitness, population):
    population = sorted(population, key=lambda chromo: chromo[0], reverse=True)
    non_zero_pop = [x[0] for x in population if x[0] > average_fitness]
    return np.mean(non_zero_pop)


def process_parallel(params):
    index, peepo, population = params
    peepo.update()
    population[index][1] = peepo.food


def main():
    # generate_food(300)
    cpu_processes = 4
    num_individuals = 30
    num_generations = 50
    ga = GeneticAlgorithm('survival',
                          min_fitness_score=0.0,
                          p_mut_top=0.2,
                          p_mut_cpd=0.2,
                          Npop=num_individuals,
                          max_removal=2)
    population = ga.get_population()
    max_age = 100
    avg_fitnesses = []
    treshold = 0

    for gen in range(num_generations):
        food = read_food()
        food.append(Food('cheat', (20, 20)))

        print('************************* GENERATION ', gen, ' *************************')

        peepos = create_population(gen, population, food)
        for age in range(max_age):
            print('-----------> AGE OF PEEPOS ', age, ' <-----------')

            ProcessPool(nodes=cpu_processes).map(
                process_parallel, [(index, peepo, population) for index, peepo in enumerate(peepos)])

        avg_fitness, population = ga.evolve(population, treshold)
        if avg_fitness < 0:
            print('population collapsed :-( ')
            break

        ''' PROPOSAL FOR NORMALIZE FITNESS FOR THIS CASE '''
        treshold = minimum_normalized_fitness_score(avg_fitness, population)

        print('Average fitness: ', avg_fitness)

        avg_fitnesses.append(avg_fitness)

    t = np.arange(0.0, len(avg_fitnesses), 1)
    f, ax = plt.subplots()
    ax.plot(t, avg_fitnesses)
    ax.set(xlabel='generation', ylabel='average fitness',
           title='Survival with genetic algorithm')
    ax.grid()
    plt.show()


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    main()
