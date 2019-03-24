import json
import logging
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pygame as pg

from peepo.playground.game_of_life.organism import Ennemies, Food, Peepo
from peepo.pp.v3.genetic_algorithm import GeneticAlgorithm
from peepo.pp.v3.peepo_network import read_from_file, write_to_file

CAPTION = "game of life"
SCREEN_SIZE = (800, 800)
SCREEN_CENTER = (400, 400)


def create_population(graphical, generation, individuals, ennemies, food):
    pop = []
    for i, idv in enumerate(individuals):
        peepo = Peepo(name='peepo_' + str(generation) + '_' + str(i),
                      network=idv[1],
                      graphical=graphical,
                      pos=(5, 400),
                      ennemies=ennemies,
                      food=food)
        pop.append(peepo)

    return pop


def generate_ennemies(num):
    objects = []
    for x in range(0, num):
        objects.append({
            'id': 'obj_' + str(x),
            'x': random.randint(20, SCREEN_SIZE[0] - 20),
            'y': random.randint(20, SCREEN_SIZE[1] - 20)
        })
    with open('ennemies.json', 'w') as outfile:
        json.dump(objects, outfile)


def read_ennemies(graphical):
    ennemies = []
    with open('ennemies.json') as json_data:
        for f in json.load(json_data):
            ennemies.append(Ennemies(f['id'], (f['x'], f['y']), graphical))
    return ennemies


def generate_food(num):
    objects = []
    for x in range(0, num):
        objects.append({
            'id': 'obj_' + str(x),
            'x': random.randint(20, SCREEN_SIZE[0] - 20),
            'y': random.randint(20, SCREEN_SIZE[1] - 20)
        })
    with open('food.json', 'w') as outfile:
        json.dump(objects, outfile)


def read_food(graphical):
    food = []
    with open('food.json') as json_data:
        for f in json.load(json_data):
            food.append(Food(f['id'], (f['x'], f['y']), graphical))
    return food


class World(object):

    def __init__(self, graphical, peepos, ennemies, food):
        if graphical:
            self.screen = pg.display.get_surface()
            self.screen_rect = self.screen.get_rect()
        self.graphical = graphical
        self.clock = pg.time.Clock()
        self.fps = 60
        self.done = False
        self.peepos = peepos
        self.ennemies = ennemies
        self.food = food
        self.traject = []
        self.eaten = []
        self.collision = []

    def event_loop(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True

    def render(self):
        self.screen.fill(pg.Color("white"))
        for obj in self.ennemies:
            obj.draw(self.screen)
        for obj in self.food:
            obj.draw(self.screen)
        for peepo in self.peepos:
            peepo.draw(self.screen)

        pg.display.update()

    def main_loop(self, max_age, verify = False):
        loop = 0
        last_food = 0
        last_collision = 0
        while not self.done:
            for peepo in self.peepos:
                peepo.update()
                self.food = peepo.food
                self.traject.append(peepo.rect)
                if peepo.stomach > last_food:
                    last_food = peepo.stomach
                    self.eaten.append(peepo.rect)
                if peepo.bang > last_collision:
                    last_collision = peepo.bang
                    self.collision.append(peepo.rect)


            if self.graphical:
                self.event_loop()
                self.render()
                self.clock.tick(self.fps)
            loop += 1
            if loop % 10 == 0:
                print('Age ' + str(loop) + ' out of ' + str(max_age))
            if loop > max_age:
                for peepo in self.peepos:
                    print('Peepo got ', peepo.stomach, ' food items  and ', peepo.bang , ' injuries.')
                break


        if self.graphical:
            self.screen.fill(pg.Color("white"))
            for obj in self.ennemies:
                obj.draw(self.screen)
            for obj in self.food:
                obj.draw(self.screen)
            if verify:
                for traj in self.traject:
                    pg.draw.circle(self.screen, (225,220,225), [int(traj.x), int(traj.y)], 2)
                for traj in self.eaten:
                    pg.draw.circle(self.screen, (0,   255,  0), [int(traj.x), int(traj.y)], 4)
                for traj in self.collision:
                    pg.draw.circle(self.screen, (255,   0,  0), [int(traj.x), int(traj.y)], 4)
            pg.display.update()

def verification(graphical):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    os.environ['SDL_VIDEO_CENTERED'] = '1'
    if graphical:
        pg.init()
        pg.display.set_caption(CAPTION)
        pg.display.set_mode(SCREEN_SIZE)

    max_age = 400#2000
    ennemies = read_ennemies(graphical)
    food = read_food(graphical)
    peepos = [Peepo('peepo', read_from_file('best_life_game_network'), graphical, (5, 400), ennemies = ennemies, food = food)]
    world = World(graphical, peepos, ennemies, food)

    world.main_loop(max_age,True)
    while True:
        a = 1
    pg.quit()
    sys.exit()


def evolution(graphical, enemy_wheight):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    if graphical:
        pg.init()
        pg.display.set_caption(CAPTION)
        pg.display.set_mode(SCREEN_SIZE)

    max_age = 400
    num_individuals = 20
    num_generations = 20

    ga = GeneticAlgorithm('game_of_life',
                          convergence_period=5,
                          convergence_sensitivity_percent=5.,
                          fast = True,
                          p_mut_top=0.2,
                          p_mut_cpd=0.2,
                          Npop=num_individuals,
                          max_removal=2)
    population = ga.get_population()

    avg_fitnesses = []
    for gen in range(num_generations):
        ennemies = read_ennemies(graphical)
        food = read_food(graphical)
        peepos = create_population(graphical, gen, population, ennemies, food)
        print('Generation ' + str(gen) + ' out of ' + str(num_generations), '  with ', len(peepos) , ' peepos')
        print('-------------------------------------------------------------------------------------------------')

        world = World(graphical, peepos, ennemies, food)
        world.main_loop(max_age)
        for idx, peepo in enumerate(peepos):

            population[idx][0] = (1.0+peepo.stomach*(1.-enemy_wheight))/(1.0+ peepo.bang*enemy_wheight)

        avg_fitness, population, converging = ga.evolve(population)
        if converging:
            break
        if avg_fitness < 0:
            print(' population collapsed :-(')
            break
        print('Average fitness: ', avg_fitness)
        print('----------------------------------------------------------')
        avg_fitnesses.append(avg_fitness)
    final_network, best_fitness = ga.get_optimal_network()
    print('\n\nFINAL NETWORK has a fitness of ', best_fitness)
    print('________________\n\n')
    print(final_network.edges)
    write_to_file('best_life_game_network', final_network)

    t = np.arange(0.0, len(avg_fitnesses), 1)
    fig, ax = plt.subplots()
    ax.plot(t, avg_fitnesses)
    ax.set(xlabel='generation', ylabel='average fitness',
           title='Game of life game with genetic algorithm')
    ax.grid()
    plt.show()


if __name__ == '__main__':
    enemy_wheight = 0.5
    # generate_ennemies(200)
    # generate_food(200)
    # evolution(False,enemy_wheight)
    verification(True)
