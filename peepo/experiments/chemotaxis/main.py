import sys, os
import pygame as pg
import logging

import matplotlib.pyplot as plt
import numpy as np

from peepo.experiments.chemotaxis.organism import Bacteria
from peepo.experiments.chemotaxis.world import World
from peepo.pp.generative_model import GenerativeModel
from peepo.pp.genetic_algorithm import GeneticAlgorithm
from peepo.pp.peepo_network import read_from_file, write_to_file

CAPTION = "Bacterial Chemotaxis"
SCREEN_SIZE = (800, 800)


def run():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    os.environ['SDL_VIDEO_CENTERED'] = '1'

    pg.init()
    pg.display.set_caption(CAPTION)
    pg.display.set_mode(SCREEN_SIZE)

    network = read_from_file('best_chemotaxis')
    bacteria = [Bacteria("e.coli", network, True, (115, 115))]

    # generative_model = GenerativeModel(Bacteria("e.coli", network, (400, 400)), n_jobs=1)
    # result = generative_model.predict()
    # print(result)
    world = World(bacteria)

    world.main_loop(100000)

    pg.quit()
    sys.exit()


def create_population(graphical, generation, individuals):
    pop = []
    for i, idv in enumerate(individuals):
        peepo = Bacteria(name='e.coli_' + str(generation) + '_' + str(i),
                              network=idv[1],
                              graphical=graphical,
                              pos=(5, 400))
        pop.append(peepo)

    return pop


def evolution(graphical):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    os.environ['SDL_VIDEO_CENTERED'] = '1'
    if graphical:
        pg.init()
        pg.display.set_caption(CAPTION)
        pg.display.set_mode(SCREEN_SIZE)

    max_age = 1000
    num_individuals = 10
    num_generations = 30

    ga = GeneticAlgorithm('chemotaxis',
                          p_mut_top=0.2,
                          p_mut_cpd=0.2,
                          Npop=num_individuals,
                          max_removal=2)
    population = ga.get_population()

    avg_fitnesses = []
    for gen in range(num_generations):
        peepos = create_population(graphical, gen, population)
        print('Generation ' + str(gen) + ' out of ' + str(num_generations), '  with ', len(peepos), ' peepos')
        print('-------------------------------------------------------------------------------------------------')

        world = World(peepos, graphical)
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
        print('----------------------------------------------------------')
        avg_fitnesses.append(avg_fitness)
    final_network, best_fitness = ga.get_optimal_network()
    print('\n\nFINAL NETWORK has a fitness of ', best_fitness)
    print('________________\n\n')
    write_to_file('best_chemotaxis', final_network)

    t = np.arange(0.0, len(avg_fitnesses), 1)
    fig, ax = plt.subplots()
    ax.plot(t, avg_fitnesses)
    ax.set(xlabel='generation', ylabel='average fitness',
           title='chemotaxis with genetic algorithm')
    ax.grid()
    plt.show()


if __name__ == '__main__':
    run()
    #evolution(graphical=True)
