import itertools
import math
import random
import uuid

import numpy as np

from peepo.evolution.mutation import add_edge, remove_edge
from peepo.evolution.utilities import get_topologies, ga_child_cpd, get_adjency_map, adjency_to_edges
from peepo.pp.peepo_network import read_from_file
from peepo.pp.utils import get_index_matrix


class Individual:

    def __init__(self, fitness=0.0, network=None):
        self.fitness = fitness
        self.network = network


class GeneticAlgorithm:
    THRESHOLD = 3
    NUMBER_OF_PARENTS_RATIO = 1. / 5.
    TOPOLOGY_MUTATIONS = [
        # TODO
        # add_node,
        add_edge,
        # remove_node,
        remove_edge
    ]

    def __init__(self, source, n_pop=1, p_mut_top=0.02, p_mut_cpd=0.02, max_removal=None, simple_start=False):
        random.seed()
        self.source = source
        self.n_pop = n_pop
        self.p_mut_top = p_mut_top
        self.p_mut_cpd = p_mut_cpd
        self.max_removal = max_removal
        self.simple_start = simple_start
        self.number_of_parents = int(self.n_pop * GeneticAlgorithm.NUMBER_OF_PARENTS_RATIO)
        self.number_of_parents = self.number_of_parents if self.number_of_parents >= 2 else 2
        self.best_chromosome = None
        self.last_generation = None
        self.fitness_history = []

    def first_generation(self):
        """
        Generates the 0th generation population.

        :return New population
        """
        peepo_template = read_from_file(self.source)
        population = []

        if not peepo_template.get_root_nodes():
            for _ in range(0, int(len(peepo_template.get_leaf_nodes()) / 3)):
                peepo_template.add_belief_node(str(uuid.uuid4())[:8], 2)

        # TODO: use get_topologies once accepted
        topologies = get_topologies(peepo_template, simple_first=self.simple_start, max_topologies=self.n_pop)
        for topology in topologies:
            peepo_template.edges = topology['edges']
            individual = peepo_template.copy()

            for node in individual.get_nodes():
                parent_nodes = individual.get_incoming_edges(node)
                if len(parent_nodes) == 0:
                    omega = []
                    cpd = np.full(individual.cardinality_map[node], 1 / individual.cardinality_map[node])
                else:
                    parents_card = [individual.cardinality_map[p] for p in parent_nodes]
                    max_omega = 2 * math.pi * np.prod(parents_card)
                    omega = np.random.rand(individual.cardinality_map[node]) * max_omega
                    cpd = ga_child_cpd(parents_card, omega)

                individual.add_cpd(node, cpd)
                individual.add_omega(node, omega)
            individual.assemble()
            population.append(Individual(0.0, individual))

        self.best_chromosome = population[0]
        self.last_generation = (0.0, population)
        return population

    def evolve(self, population):
        """
        Evolves a given population by selection, cross-over and mutation.
            - selection: selects the best individuals from the population to evolve from
            - cross-over: generate the offspring based on the selected individuals
            - mutation: mutate parameters and structure of the bayesian network of each individual offspring

        :param population to evolve
        :return Average Fitness
        :return New population with mutated offspring
        """
        population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
        # Best Chromosome yet
        if population[0].fitness >= self.best_chromosome.fitness:
            self.best_chromosome = population[0]

        # Average fitness is calculated on top performing individuals
        avg_fitness = 0
        for i in range(self.number_of_parents):
            avg_fitness += population[i].fitness
            avg_fitness /= self.number_of_parents

        # SELECTION
        selected_parents = self.get_selected_parents(population, avg_fitness)
        # CROSS-OVER
        selected_offspring = self.cross_over(selected_parents)
        # MUTATION
        selected_offspring = self.mutate(selected_offspring)

        # Collect parents and offspring
        random.shuffle(selected_offspring)
        new_population = []

        for offspring in selected_offspring:
            new_population.append(offspring[0])
        for parent in selected_parents:
            new_population.append(parent)
            if len(new_population) >= self.n_pop:
                break

        return avg_fitness, new_population

    def get_optimal_network(self):
        """
        Returns the best network over all generation with it's fitness score
        :return Individual
        """
        return self.best_chromosome

    def mutate(self, selected_offspring):
        for offspring in selected_offspring:
            mut_top = offspring[1]
            mut_cpd = offspring[2]

            if mut_top <= self.p_mut_top:
                offspring[0].network = self.mutate_topology(offspring[0].network)
            if mut_cpd <= self.p_mut_cpd:
                offspring[0].network = self.mutate_cpds(offspring[0].network)

        return selected_offspring

    def cross_over(self, selected_parents):
        selected_offspring = []
        mating_couples = list(itertools.combinations(selected_parents, 2))

        # To control exponential growth, limit number of combinations
        if len(mating_couples) > self.n_pop:
            random.shuffle(mating_couples)
            mating_couples = mating_couples[0:int(self.n_pop / 2.)]

        for n, chrom in enumerate(mating_couples):
            map_1 = get_adjency_map(chrom[0].network)
            map_2 = get_adjency_map(chrom[1].network)
            diff = np.abs(map_1 - map_2)
            i_sum = np.sum(diff)

            if i_sum == 0 or i_sum > GeneticAlgorithm.THRESHOLD:
                selected_offspring.append((Individual(0.0, chrom[0].network.copy()), 0, 0))
                selected_offspring.append((Individual(0.0, chrom[1].network.copy()), 0, 0))
                continue

            indices = np.argwhere(diff == 1)
            combinations = [[1, 0], [0, 1]]
            if len(indices) > 1:
                combinations = np.transpose(get_index_matrix(np.full(len(indices), 2).tolist()))

            for comb in combinations:
                i_map = np.copy(map_1)
                for pos, index in enumerate(indices):
                    i_map[index[0], index[1]] = comb[pos]

                if 0 in np.sum(i_map, axis=1) or np.array_equal(i_map, map_1) or np.array_equal(i_map, map_2):
                    continue

                edges = adjency_to_edges(chrom[0].network, i_map)
                new_peepo = chrom[0].network.copy()
                new_peepo.disassemble()
                new_peepo.edges = edges
                for node in new_peepo.get_nodes():
                    incoming_nodes = new_peepo.get_incoming_edges(node)
                    if len(incoming_nodes) == 0:
                        my_cpd = np.full(new_peepo.cardinality_map[node], 1. / new_peepo.cardinality_map[node])
                        my_omega = []
                    else:
                        my_card_parents = []
                        [my_card_parents.append(new_peepo.cardinality_map[nod]) for nod in incoming_nodes]
                        max_omega = 2 * math.pi * np.prod(my_card_parents)
                        my_omega = np.random.rand(new_peepo.cardinality_map[node]) * max_omega
                        my_cpd = ga_child_cpd(my_card_parents, my_omega)
                    new_peepo.add_cpd(node, my_cpd)
                    new_peepo.add_omega(node, my_omega)
                new_peepo.assemble()

                if np.array_equal(comb[0], comb[1]):
                    selected_offspring.append((Individual(0.0, new_peepo), 0, 0))
                else:
                    selected_offspring.append((Individual(0.0, new_peepo),
                                               random.uniform(0, 1),
                                               random.uniform(0, 1)))

        # If there's not enough offspring, fill it up with parents
        while True:
            if len(selected_parents) + len(selected_offspring) >= self.n_pop:
                break
            for parent in selected_parents:
                if len(selected_parents) + len(selected_offspring) >= self.n_pop:
                    break
                selected_offspring.append((parent, 0, 0))

        return selected_offspring

    @staticmethod
    def mutate_topology(network):
        return random.choice(GeneticAlgorithm.TOPOLOGY_MUTATIONS)(network)

    @staticmethod
    def mutate_cpds(network):
        epsilon = random.uniform(0.05, 0.75)
        for leaf in network.get_leaf_nodes():
            parents_card = [network.cardinality_map[x] for x in network.get_incoming_edges(leaf)]
            network.omega_map[leaf] += (0.5 - np.random.rand(network.cardinality_map[leaf])) * epsilon
            network.add_cpd(leaf, ga_child_cpd(parents_card, network.omega_map[leaf]))
        return network

    def get_selected_parents(self, population, avg_fitness):
        selected_parents = []

        pool = []
        pop_len = len(population)
        for i in range(pop_len):
            repeat = pop_len - i
            for rep in range(repeat):
                pool.append(i)
        random.shuffle(pool)
        for draw in range(self.number_of_parents):
            parent_index = pool[random.randint(0, len(pool) - 1)]
            selected_parents.append(Individual(0.0, population(parent_index).network.copy()))
        random.shuffle(selected_parents)

        prev_fitness = self.last_generation[0]

        if avg_fitness < prev_fitness:
            avg_fitness = (prev_fitness + avg_fitness) / 2
            new_parents = self.last_generation[1] + selected_parents
            random.shuffle(new_parents)
            selected_parents = new_parents[0:self.number_of_parents]

        self.last_generation = (avg_fitness, selected_parents)

        return selected_parents
