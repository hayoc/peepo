import json
import random

from peepo.playground.survival.organism import Peepo, Food

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
    with open('food_stuff.json') as json_data:
        for f in json.load(json_data):
            food_stuff.append(Food(f['name'], (f['x'], f['y'])))
    return food_stuff


def create_population(generation, individuals, ancestors, food):
    pop = []

    if generation == 0:
        for idv in range(0, individuals):
            peepo = Peepo(name='peepo_' + str(generation) + '_' + str(idv),
                          network=None,  # TODO: Get network from GA
                          pos=(5, 5),
                          obstacles=food)
            pop.append(peepo)
    else:
        for idv in range(0, individuals):
            peepo = Peepo(name='peepo_' + str(generation) + '_' + str(idv),
                          network=None,  # TODO: Get network from GA
                          pos=(5, 5),
                          obstacles=food)
            pop.append(peepo)

    return pop


if __name__ == '__main__':
    # generate_food(30)

    num_individuals = 10
    num_generations = 100
    population = []
    for gen in range(0, num_generations):
        food = read_food()
        population = create_population(gen, num_individuals, population, food)
