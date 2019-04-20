import logging

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import logger

from peepo.pp.generative_model import GenerativeModel
from peepo.pp.genetic_algorithm import GeneticAlgorithm
from peepo.pp.peepo import Peepo
from peepo.pp.peepo_network import write_to_file

VISION = 'vision'
MOTOR = 'motor'


class PeepoAgent(Peepo):
    def __init__(self, network, action_space, observation_space):
        super().__init__(network)
        self.action_space = action_space
        self.observation_space = observation_space
        self.obs = np.empty(4)
        self.reward = 0
        self.done = False
        self.act = 0
        self.generative_model = GenerativeModel(self, n_jobs=4)

    def update(self, obz, rewardz, donez):
        self.obs = obz
        self.reward = rewardz
        self.done = donez

        self.generative_model.process()

        return self.act

    def observation(self, name):
        if VISION.lower() in name.lower():
            quad = self.get_quadrant(name)
            return self.normalized_distribution(self.obs[quad],
                                                self.observation_space.low[quad],
                                                self.observation_space.high[quad])
        if MOTOR.lower() in name.lower():
            return [0.1, 0.9] if self.act else [0.9, 0.1]

        logging.warning('Reached code which should not be reached in observation')
        return [0.5, 0.5]

    def action(self, node, prediction):
        self.act = np.argmax(prediction)

    @staticmethod
    def normalized_distribution(value, mini, maxi, target_min=0, target_max=1):
        if str(maxi) == '3.4028235e+38':
            mini, maxi = -1, 1
        x = target_max * ((value - mini) / (maxi - mini)) + target_min
        return np.array([x, 1 - x])

    @staticmethod
    def get_quadrant(name):
        for quad in ['0', '1', '2', '3']:
            if quad.lower() in name.lower():
                return int(quad)
        raise ValueError('Unexpected node name %s, could not find 0,1,2,3', name)


if __name__ == '__main__':
    logger.set_level(logger.INFO)

    env = gym.make('CartPole-v1')
    # env = gym.make('BipedalWalker-v2')

    env.seed(0)

    reward = 0
    done = False

    max_age = 100
    num_individuals = 10
    num_generations = 30
    ga = GeneticAlgorithm('cartpole',
                          p_mut_top=0.2,
                          p_mut_cpd=0.2,
                          Npop=num_individuals,
                          max_removal=2)
    population = ga.get_population()

    avg_fitnesses = []

    for gen in range(num_generations):

        for i, idv in enumerate(population):
            agent = PeepoAgent(idv[1], env.action_space, env.observation_space)

            ob = env.reset()
            while True:
                action = agent.update(ob, reward, done)
                ob, reward, done, _ = env.step(action)
                # env.render()
                if done:
                    population[i][0] = reward
                    ob = env.reset()
                    break

        avg_fitness, population, converging = ga.evolve(population)
        if avg_fitness < 0:
            print(' population collapsed :-(')
            break

        print('Generation: ' + str(gen) + ' , Average fitness: ' + avg_fitness)
        avg_fitnesses.append(avg_fitness)

    env.close()

    final_network, best_fitness = ga.get_optimal_network()
    write_to_file('best_cartpole_network', final_network)

    t = np.arange(0.0, len(avg_fitnesses), 1)
    fig, ax = plt.subplots()
    ax.plot(t, avg_fitnesses)
    ax.set(xlabel='generation', ylabel='average fitness',
           title='Survival with genetic algorithm')
    ax.grid()
    plt.show()
