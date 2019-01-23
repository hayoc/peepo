import json
import logging
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pygame as pg

from peepo.playground.wandering.organism import Obstacle, Peepo
from peepo.predictive_processing.v3.genetic_algorithm import GeneticAlgorithm
from peepo.predictive_processing.v3.peepo_network import read_from_file, write_to_file

CAPTION = "Survival"
SCREEN_SIZE = (800, 800)
SCREEN_CENTER = (400, 400)


def create_population(graphical, generation, individuals, food):
    pop = []
    for i, idv in enumerate(individuals):
        peepo = Peepo(name='peepo_' + str(generation) + '_' + str(i),
                      network=idv[1],
                      graphical=graphical,
                      pos=(5, 5),
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
            print('Age ' + str(loop) + ' out of ' + str(max_age))
            if loop > max_age:
                break


def graphical_run():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    # generate_obstacles(400)
    graphical = True
    max_age = 500

    os.environ['SDL_VIDEO_CENTERED'] = '1'
    if graphical:
        pg.init()
        pg.display.set_caption(CAPTION)
        pg.display.set_mode(SCREEN_SIZE)

    obstacles = read_obstacles(graphical)
    peepos = [Peepo('peepo', read_from_file('wandering'), graphical, (5, 400), obstacles)]
    world = World(graphical, peepos, obstacles)

    world.main_loop(max_age)
    print(peepos[0].health)

    pg.quit()
    sys.exit()


def non_graphical_run():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    # generate_obstacles(400)
    graphical = False

    max_age = 500
    num_individuals = 5
    num_generations = 2

    ga = GeneticAlgorithm('wandering',
                          convergence_period=10,
                          convergence_sensitivity_percent=5.,
                          min_fitness_score=0.0,
                          p_mut_top=0.2,
                          p_mut_cpd=0.2,
                          Npop=num_individuals,
                          max_removal=2)
    population = ga.get_population()

    avg_fitnesses = []
    for gen in range(num_generations):
        print('Generation ' + str(gen) + ' out of ' + str(num_generations))
        print('----------------------------------------------------------')
        obstacles = read_obstacles(graphical)
        peepos = create_population(graphical, gen, population, obstacles)

        world = World(graphical, peepos, obstacles)
        world.main_loop(max_age)
        for idx, peepo in enumerate(peepos):
            population[idx][0] = peepo.health

        avg_fitness, population, converging = ga.evolve(population)
        if converging:
            break
        if avg_fitness < 0:
            print(' population collapsed :-(')
            break
        print('Average fitness: ', avg_fitness)
        avg_fitnesses.append(avg_fitness)
    final_network, best_fitness = ga.get_optimal_network()
    print('\n\nFINAL NETWORK has a fitness of ', best_fitness)
    print('________________\n\n')
    print(final_network.edges)
    write_to_file('best_wandering_network', final_network)

    t = np.arange(0.0, len(avg_fitnesses), 1)
    fig, ax = plt.subplots()
    ax.plot(t, avg_fitnesses)
    ax.set(xlabel='generation', ylabel='average fitness',
           title='Survival with genetic algorithm')
    ax.grid()
    plt.show()


if __name__ == '__main__':
    non_graphical_run()
