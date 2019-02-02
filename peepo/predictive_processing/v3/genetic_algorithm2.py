import itertools
import logging
import math
import random
import uuid

import numpy as np

from peepo.predictive_processing.v3.peepo_network import read_from_file, get_topologies2


def mutate_topology(network):
    return random.choice(GeneticAlgorithm.TOPOLOGY_MUTATIONS)(network)


def mutate_cpds(network):
    epsilon = random.uniform(0.05, 0.75)
    for leaf in network.get_leaf_nodes():
        parents_card = [network.cardinality_map[x] for x in network.get_incoming_edges(leaf)]
        network.omega_map[leaf] += (0.5 - np.random.rand(network.cardinality_map[leaf])) * epsilon
        network.add_cpd(leaf, ga_child_cpd(parents_card, network.omega_map[leaf]))
    return network


def add_node(network):
    """
    Random mutation of adding a single node and an edge.
    :param network: PeepoNetwork to mutate
    :return: mutated PeepoNetwork
    """
    new_parent_node = str(uuid.uuid4())[:8]
    child_node = random.choice(network.get_leaf_nodes())
    logging.info('Adding new node with edge to: %s to %s', new_parent_node, child_node)

    network.add_belief_node(new_parent_node, 2)
    network.add_edge((new_parent_node, child_node))

    parents_card = [network.cardinality_map[x] for x in network.get_incoming_edges(child_node)]

    network.add_omega(new_parent_node, np.random.rand(2) * (2 * math.pi))
    network.add_cpd(new_parent_node, ga_parent_cpd(network.omega_map[new_parent_node]))
    network.add_cpd(child_node, ga_child_cpd(parents_card, network.omega_map[child_node]))

    return network


def add_edge(network):
    """
    Random mutation of adding a single edge between two random chosen nodes.
    :param network: PeepoNetwork to mutate
    :return: mutated PeepoNetwork
    """
    node_pool = network.get_root_nodes()
    if not node_pool:
        logging.debug('Model contains no valid nodes to add edge from... Adding a new node')
        return add_node(network)

    parent_node = random.choice(node_pool)
    child_pool = np.setdiff1d(network.get_leaf_nodes(), network.get_outgoing_edges(parent_node), assume_unique=True)
    if not child_pool.size:
        return remove_edge(network)
    child_node = random.choice(child_pool)
    logging.info('Adding edge from %s to %s', parent_node, child_node)

    network.add_edge((parent_node, child_node))

    parents_card = [network.cardinality_map[x] for x in network.get_incoming_edges(child_node)]

    network.add_cpd(child_node, ga_child_cpd(parents_card, network.omega_map[child_node]))

    return network


def remove_node(network):
    """
    Random mutation of removing a single node chosen randomly.
    :param network: PeepoNetwork to mutate
    :return: mutated PeepoNetwork
    """
    node_pool = network.get_root_nodes()
    if not node_pool:
        logging.debug('Model contains no valid nodes to remove... Adding a new node')
        return add_node(network)

    to_remove = random.choice(node_pool)
    affected_children = network.get_outgoing_edges(to_remove)

    network.remove_belief_node(to_remove)
    logging.info('Removed node %s', to_remove)

    for child_node in affected_children:
        parents_card = [network.cardinality_map[x] for x in network.get_incoming_edges(child_node)]

        network.add_cpd(child_node, ga_child_cpd(parents_card, network.omega_map[child_node]))

    return network


def remove_edge(network):
    """
    Random mutation of removing a single edge between two random chosen nodes.
    :param network: PeepoNetwork to mutate
    :return: mutated PeepoNetwork
    """
    node_pool = network.get_root_nodes()
    if not node_pool:
        logging.debug('Model contains no valid nodes to remove an edge from... Adding a new node')
        return add_node(network)

    parent_node = random.choice(node_pool)
    outgoing_edges = network.get_outgoing_edges(parent_node)
    if len(outgoing_edges) <= 1:
        logging.debug('Model contains no valid edges to remove... Adding a new edge instead')
        return add_edge(network)

    child_node = random.choice(outgoing_edges)

    network.remove_edge((parent_node, child_node))
    logging.info('Removed edge (%s, %s)', parent_node, child_node)

    parents_card = [network.cardinality_map[x] for x in network.get_incoming_edges(child_node)]

    network.add_cpd(child_node, ga_child_cpd(parents_card, network.omega_map[child_node]))

    return network


def ga_child_cpd(card_parents, omega):
    """
    Used in the framework of Genetic Algorithm to initialize a childs cpd and to alter it later (mutation)
    The word 'child" used here is in the context of Bayesian network and is not an offspring of a GA


    :param card_parents: array containing the cardinality of the childs' parents
    :param omega: an array of dim cardinality of the child (card_child) containing the cpd's  generating parameters
    :return: a matrix of shape (card_child, numpy.prod(card_parents) containing the conditional probability distribution

    example :

        my_card_parents = [2, 3, 2]
        max_omega = 2 * math.pi * np.prod(my_card_parents)

        zero generation : (my_card_child = 3)
        my_omega = [2.0,0.1,0.9] -> only for this example. As a rule one will use
            my_omega = np.random.rand(my_card_child) * max_omega to initialize (0th generation)
            a child's cpd and
            my_omega += (0.5 - np.random.rand(my_card_child))*epsilon with espilon small (e.g. 0.05)
            when mutating the childs' cpd

        my_pdf = ga_child_cpd( my_card_parents, my_omega)

        mutation :
        epsilon = 0.05
        my_omega += (0.5 - np.random.rand(len(my_omega))*epsilon
        my_pdf_ = ga_child_cpd( my_card_parents, my_omega)

        --->
            Zero generation cpd
            [[0.07 0.25 0.53 0.18 0.14 0.37 0.18 0.09 0.54 0.46 0.06 0.45]
             [0.56 0.62 0.47 0.67 0.49 0.28 0.35 0.48 0.35 0.54 0.69 0.24]
             [0.37 0.13 0.00 0.15 0.37 0.35 0.47 0.44 0.11 0.00 0.25 0.31]]


            Mutated generation cpd
            [[0.08 0.23 0.54 0.21 0.13 0.37 0.20 0.06 0.55 0.55 0.03 0.47]
             [0.55 0.62 0.46 0.65 0.50 0.27 0.32 0.45 0.32 0.45 0.71 0.21]
             [0.37 0.14 0.00 0.13 0.38 0.36 0.47 0.49 0.13 0.00 0.26 0.33]]


            Delta zero generation - mutated cpd
            [[-0.01 0.01 -0.01 -0.03 0.02 -0.01 -0.03 0.03 -0.01 -0.08 0.04 -0.02]
             [0.01 -0.00 0.01 0.02 -0.01 0.01 0.03 0.03 0.03 0.09 -0.03 0.03]
             [-0.00 -0.01 -0.00 0.01 -0.01 -0.00 -0.00 -0.05 -0.03 -0.00 -0.01 -0.01]]
    """

    phase_shift = omega[0]
    n_comb = int(np.prod(card_parents))
    pdf = []
    for ang in omega:
        pdf_row = []
        for col in range(n_comb):
            pdf_row.append(math.sin(ang * (col + 1) + phase_shift) + 1.2)
        pdf.append(pdf_row)
    return normalize_distribution(np.asarray(pdf))


def ga_parent_cpd(omega):
    phase_shift = omega[0]

    pdf = []
    for idx, ang in enumerate(omega):
        pdf.append(math.sin(ang * (idx + 1) + phase_shift) + 1.2)

    return normalize_distribution(np.asarray(pdf))


def normalize_distribution(matrix):
    """
    Normalizes the columns of a matrix (i.e. sum matrix[:,i] = 1

    :param matrix: numpy 2 dimensional array
    :return: numpy 2 dimensional normalized array
    """
    factor = np.sum(matrix, axis=0)
    return matrix / factor


class Individual:

    def __init__(self, fitness=0.0, network=None):
        self.fitness = fitness
        self.network = network


class GeneticAlgorithm:
    NUMBER_OF_PARENTS_RATIO = 1. / 5.
    TOPOLOGY_MUTATIONS = [
        # TODO
        # add_node,
        add_edge,
        # remove_node,
        remove_edge
    ]

    def __init__(self, source, fast_convergence=False, convergence_period=100000, convergence_sensitivity=5.,
                 n_pop=1, p_mut_top=0.02, p_mut_cpd=0.02, max_removal=None, simple_start=False):
        random.seed()
        self.source = source
        self.fast_convergence = fast_convergence
        self.convergence_period = convergence_period
        self.convergence_sensitivity = convergence_sensitivity
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
        peepo_template = read_from_file(self.source)
        population = []

        if not peepo_template.get_root_nodes():
            for _ in range(0, int(len(peepo_template.get_leaf_nodes()) / 3)):
                peepo_template.add_belief_node(str(uuid.uuid4())[:8], 2)

        # TODO: use get_topologies once accepted
        topologies = get_topologies2(peepo_template, simple_first=self.simple_start, max_topologies=self.n_pop)
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
        population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
        # Best Chromosome yet
        if population[0].fitness >= self.best_chromosome.fitness:
            self.best_chromosome = population[0]

        # Average fitness is calculated on top performing individuals
        avg_fitness = 0
        for i in range(self.number_of_parents):
            avg_fitness += population[i].fitness
            avg_fitness /= self.number_of_parents

        # TODO
        # Stop if fitness converges across defined period
        # convergence = self.check_convergence(avg_fitness)
        # if convergence:
        #     return avg_fitness, population, True

        # SELECTION: Get parents which will reproduce
        selected_parents = self.get_selected_parents(population, avg_fitness)

        # CROSS-OVER: Get offspring via cross-over TODO
        selected_offspring = self.cross_over(selected_parents)

        # MUTATION: Mutate resulting offspring
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

        return avg_fitness, new_population, False

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
                offspring[0].network = mutate_topology(offspring[0].network)
            if mut_cpd <= self.p_mut_cpd:
                offspring[0].network = mutate_cpds(offspring[0].network)

        return selected_offspring

    def cross_over(self, selected_parents):
        selected_offspring = []
        mating_couples = list(itertools.combinations(selected_parents, 2))

        # To control exponential growth, limit number of combinations
        if len(mating_couples) > self.n_pop:
            random.shuffle(mating_couples)
            mating_couples = mating_couples[0:int(self.n_pop / 2.)]

        for n, chrom in enumerate(mating_couples):
            selected_offspring.append((Individual(0.0, chrom[0].network.copy()),
                                       random.uniform(0, 1),
                                       random.uniform(0, 1)))  # TODO
            selected_offspring.append((Individual(0.0, chrom[1].network.copy()),
                                       random.uniform(0, 1),
                                       random.uniform(0, 1)))  # TODO

        return selected_offspring

    def get_selected_parents(self, population, avg_fitness):
        selected_parents = []

        if self.fast_convergence:
            for i in range(self.number_of_parents):
                selected_parents.append(Individual(0.0, population[i].network.copy()))
            random.shuffle(selected_parents)
        else:
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

    def check_convergence(self, avg_fitness):
        if len(self.fitness_history) <= self.convergence_period:
            self.fitness_history.append(avg_fitness)
            return False
        else:
            self.fitness_history.append(avg_fitness)
            self.fitness_history.pop(0)

            mean = np.mean(self.fitness_history)
            std = np.std(self.fitness_history)
            acceptable_std = mean * self.convergence_sensitivity / 2

            return std < acceptable_std
