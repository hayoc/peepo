import json
import logging
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pygame as pg

from peepo.evolution.genetic_algorithm import GeneticAlgorithm
from peepo.playground.survival.organism import Obstacle, SurvivalPeepo
from peepo.pp.peepo_network import read_from_file, write_to_file
from peepo.visualize.graph import draw_network

CAPTION = "survival"
SCREEN_SIZE = (800, 800)
SCREEN_CENTER = (400, 400)


def create_population(graphical, generation, individuals, food):
    pop = []
    for i, idv in enumerate(individuals):
        peepo = SurvivalPeepo(name='peepo_' + str(generation) + '_' + str(i),
                              network=idv.network,
                              graphical=graphical,
                              pos=(5, 400),
                              obstacles=food)
        pop.append(peepo)

    return pop


def generate_obstacles(num):
    objects = []
    for x in range(0, num):
        objects.append({
            'id': 'obj_' + str(x),
            'x': random.randint(20, SCREEN_SIZE[0] - 20),
            'y': random.randint(20, SCREEN_SIZE[1] - 20)
        })
    with open('obstacles.json', 'w') as outfile:
        json.dump(objects, outfile)


def read_obstacles(graphical):
    obstacles = []
    with open('obstacles.json') as json_data:
        for f in json.load(json_data):
            obstacles.append(Obstacle(f['id'], (f['x'], f['y']), graphical))
    return obstacles


class World(object):

    def __init__(self, graphical, peepos, obstacles):
        if graphical:
            self.screen = pg.display.get_surface()
            self.screen_rect = self.screen.get_rect()
        self.graphical = graphical
        self.clock = pg.time.Clock()
        self.fps = 60
        self.done = False
        self.peepos = peepos
        self.obstacles = obstacles

    def event_loop(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True

    def render(self):
        self.screen.fill(pg.Color("white"))
        for obj in self.obstacles:
            obj.draw(self.screen)
        for peepo in self.peepos:
            peepo.draw(self.screen)

        pg.display.update()

    def main_loop(self, max_age):
        loop = 0
        while not self.done:
            for peepo in self.peepos:
                peepo.update()
            if self.graphical:
                self.event_loop()
                self.render()
                self.clock.tick(self.fps)
            loop += 1
            # if loop % 10 == 0:
            #     print('Age ' + str(loop) + ' out of ' + str(max_age))
            if loop > max_age:
                for peepo in self.peepos:
                    print(peepo.health)
                break


def verification(graphical):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    generate_obstacles(500)

    os.environ['SDL_VIDEO_CENTERED'] = '1'
    if graphical:
        pg.init()
        pg.display.set_caption(CAPTION)
        pg.display.set_mode(SCREEN_SIZE)

    max_age = 100000
    obstacles = read_obstacles(graphical)
    peepo_network = read_from_file('best_survival_network')
    draw_network(peepo_network)

    peepos = [SurvivalPeepo('peepo', peepo_network, graphical, (5, 400), obstacles)]
    world = World(graphical, peepos, obstacles)

    world.main_loop(max_age)

    pg.quit()
    sys.exit()


def evolution(graphical):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    # generate_obstacles(400)

    os.environ['SDL_VIDEO_CENTERED'] = '1'
    if graphical:
        pg.init()
        pg.display.set_caption(CAPTION)
        pg.display.set_mode(SCREEN_SIZE)

    max_age = 400
    num_individuals = 20
    num_generations = 20

    ga = GeneticAlgorithm('survival',
                          p_mut_top=0.2,
                          p_mut_cpd=0.2,
                          n_pop=num_individuals,
                          max_removal=2)
    population = ga.first_generation()

    avg_fitnesses = []
    for gen in range(num_generations):
        obstacles = read_obstacles(graphical)
        peepos = create_population(graphical, gen, population, obstacles)
        print('Generation ' + str(gen) + ' out of ' + str(num_generations), '  with ', len(peepos), ' peepos')
        print('-------------------------------------------------------------------------------------------------')

        world = World(graphical, peepos, obstacles)
        world.main_loop(max_age)
        for idx, peepo in enumerate(peepos):
            population[idx].fitness = peepo.health

        avg_fitness, population = ga.evolve(population)

        if avg_fitness < 0:
            print(' population collapsed :-(')
            break
        print('Average fitness: ', avg_fitness)
        print('----------------------------------------------------------')
        avg_fitnesses.append(avg_fitness)
    best_individual = ga.get_optimal_network()
    print('\n\nFINAL NETWORK has a fitness of ', best_individual.fitness)
    print('________________\n\n')
    print(best_individual.network.edges)
    write_to_file('best_survival_network', best_individual.network)

    t = np.arange(0.0, len(avg_fitnesses), 1)
    fig, ax = plt.subplots()
    ax.plot(t, avg_fitnesses)
    ax.set(xlabel='generation', ylabel='average fitness',
           title='Survival with genetic algorithm')
    ax.grid()
    plt.show()


if __name__ == '__main__':
    # evolution(False)
    verification(True)
