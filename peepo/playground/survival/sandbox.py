import json
import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import copy

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







if __name__ == '__main__':
    # generate_food(300)
    # generate_food(3000)

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    num_individuals = 20
    num_generations = 10
    ga = GeneticAlgorithm('survival', min_fitness_score = 0.0, p_mut_top = 0.2, p_mut_cpd = 0.2,Npop=num_individuals, max_removal=2)
    population = ga.get_population()
    peepos = []
    max_age = 30#50
    avg_fitnesses = []
    final_population = []
    for gen in range(num_generations):
        food = read_food()
        food.append(Food('cheat', (20, 20)))
        #
        # logging.info('*********************                     GENERATION ', gen,
        #              ' *****************************************')
        peepos = create_population(gen, population, food)
        for age in range(max_age):
            # logging.info('**  GENERATION ' ,gen , ' -----------> AGE OF PEEPOS ' ,  age, ' --------------')
            print('**  GENERATION ' ,gen , ' -----------> AGE OF THE ',len(population) , ' PEEPOS = ' ,  age, ' --------------')
            final_population = []
            for ind, peepo in enumerate(peepos):
                peepo.update()
                population[ind][0] = peepo.food
                final_population.append([peepo.food, population[ind][1]])
        avg_fitness, population = ga.evolve(population)
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
    my_json = final_network.to_json()
    with open('best_survival_network.json', 'w') as outfile:
        json.dump(my_json, outfile)
    t = np.arange(0.0, len(avg_fitnesses), 1)
    fig, ax = plt.subplots()
    ax.plot(t, avg_fitnesses)
    ax.set(xlabel='generation', ylabel='average fitness',
           title='Survival with genetic algorithm')
    ax.grid()
    plt.show()
