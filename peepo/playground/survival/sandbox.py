import json
import random

import matplotlib.pyplot as plt
import numpy as np

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
                      network=idv[2],
                      pos=(5, 5),
                      obstacles=food)
        pop.append(peepo)

    return pop


if __name__ == '__main__':
    # generate_food(30)

    num_individuals = 5
    num_generations = 100
    ga = GeneticAlgorithm('survival', Npop=num_individuals, max_removal=2)
    population = ga.get_population()
    peepos = []
    max_age = 100
    avg_fitnesses = []
    for gen in range(num_generations):
        food = read_food()
        print('*********************                     GENERATION ', gen,
              ' *****************************************')
        peepos = create_population(gen, population, food)
        for age in range(max_age):
            print(' ----------- AGE OF PEEPOS ', age, ' --------------')
            for ind, peepo in enumerate(peepos):
                peepo.update()
                population[ind][1] = peepo.food
        avg_fitness, population = ga.evolve(population)
        print(avg_fitness)
        avg_fitnesses.append(avg_fitness)

    t = np.arange(0.0, len(avg_fitnesses), 1)
    ax = plt.subplots()
    ax.plot(t, avg_fitnesses)
    ax.set(xlabel='generation', ylabel='average fitness',
           title='Survival with genetic algorithm')
    ax.grid()
    plt.show()
