#18/01/2019
import json
import random
import math

import numpy as np

from config import ROOT_DIR
import copy
from itertools import combinations_with_replacement

from peepo.predictive_processing.v3.peepo_network import PeepoNetwork
from peepo.predictive_processing.v3.peepo_network import get_topologies
from peepo.predictive_processing.v3.utils import get_index_matrix

NUMBER_OF_PARENTS_RATIO = 1./5.

class GeneticAlgorithm:

    def __init__(self, source, fast = False, convergence_period = 100000, convergence_sensitivity_percent = 5., Npop = 1,  p_mut_top = 0.02, p_mut_cpd = 0.02, max_removal=None):
        self.fast = fast
        self.npop = int(Npop)
        self.population = []
        self.selected_parents = []
        self.number_of_parents = int(self.npop*NUMBER_OF_PARENTS_RATIO)
        self.population_fitness_score = 0#min_fitness_score#controls how the best parents are selected (fitness > min_fitness_score)
        self.p_mut_pop = p_mut_top
        self.p_mut_cpd = p_mut_cpd
        self.root_nodes = []
        self.leaf_nodes = []
        self.cardinality_map = {}
        self.peepo = PeepoNetwork()
        self.max_removal = max_removal
        self.treshold = 3#controls the allowed distance of the parents (i.e. how much edges they differ from each other)
        self.best_chromosome = None
        self.last_generation= [0.0,[]]
        self.convergence_period  = convergence_period
        self.convergence_history = []
        self.convergence_sensitivity = convergence_sensitivity_percent/100.
        self.initialize(source)




    def get_population(self):
        self.best_chromosome = self.population[0]
        self.last_generation= [0.0, copy.copy(self.population)]
        return self.population


    def initialize(self, source):
        """
                 Initializes the object members and generates the 0th generation  population array

                 :param source: a string of the form 'filename' with_out extension (.json expected)

                 :return :  class object

         """
        with open(ROOT_DIR + '/resources/' + source + '.json') as json_data:
            json_object = json.load(json_data)
        self.peepo.from_json(json_object)
        self.root_nodes = self.peepo.get_root_nodes()
        self.leaf_nodes = self.peepo.get_leaf_nodes()
        self.cardinality_map = self.peepo.make_cardinality_map()
        topologies = get_topologies(self.peepo, max_removal = self.max_removal)
        random.seed()
        ''' indexes is a temporary array containing Npop indexes of the topologies,
            the indexes are linearly distributed according to 
               index[i] = slope*i  with slope being calculated so only the upper half
                of topologies are considered (i.e. we discard the too simple topologies)'''
        indexes = []
        slope = 1
        if self.npop < len(topologies)/2:
            slope = int(len(topologies)/2/self.npop)
        if len(topologies) <= self.npop:
            self.npop = len(topologies)
            self.number_of_parents = int(self.npop*NUMBER_OF_PARENTS_RATIO)
        [indexes.append(slope*x) for x in range(0, self.npop)]
        #browse through the selected topologies and generate the 0th population
        for t in indexes:
            self.peepo.edges = topologies[t]['edges']
            for node in self.peepo.get_nodes():
                incoming_nodes = self.peepo.get_incoming_edges(node)
                if len(incoming_nodes)== 0 :
                    my_cpd = np.full(self.cardinality_map[node], 1/self.cardinality_map[node])
                    my_omega = []
                else:
                    my_card_parents = []
                    [my_card_parents.append(self.cardinality_map[nod]) for nod in incoming_nodes]
                    max_omega = 2 * math.pi * np.prod(my_card_parents)
                    my_omega = np.random.rand(self.cardinality_map[node]) * max_omega
                    my_cpd = self.ga_child_cpd(my_card_parents, my_omega)
                self.peepo.add_cpd(node, my_cpd)
                self.peepo.add_omega(node,my_omega)
            self.peepo.assemble()
            my_chromosome = [0, copy.copy(self.peepo)]
            self.population.append(my_chromosome)
            #cleaning
            self.peepo.disassemble()
            if self.number_of_parents ==0:
                self.number_of_parents = 1

    def get_parents(self):
        """        Selects which parent will ba allowed to create offsprings

                :param object:

                :return :  None

        """
        self.selected_parents = []
        self.population = sorted(self.population, key=lambda chromo: chromo[0], reverse=True)
        # check for the best chromosome yet
        if self.population[0][0] >= self.best_chromosome[0]:
            self.best_chromosome = [self.population[0][0], copy.copy(self.population[0][1])]
        #calculate the fitness score over the best (len(self.number_of_parents)) best performing chromosomes
        self.population_fitness_score = 0
        for i in range(self.number_of_parents):
            self.population_fitness_score += self.population[i][0]
        self.population_fitness_score /= self.number_of_parents

        if self.fast:

            self.selected_parents = self.population[0:self.number_of_parents]
            for i in range(self.number_of_parents):
                self.selected_parents[i][0] = 0.0
            self.selected_parents = self.copy_parents(self.selected_parents)
            random.shuffle(self.selected_parents)

        else:
            pool = []
            npop = len(self.population)
            for index in range(npop):
                repeat = npop - index
                [pool.append(index) for rep in range(repeat)]
            random.shuffle(pool)
            for draw in range(self.number_of_parents):
                pool_index = random.randint(0, len(pool)-1)
                parent_index = pool[pool_index]
                self.population[parent_index][0] = 0
                self.selected_parents.append([0,copy.copy(self.population[parent_index][1])])
            random.shuffle(self.selected_parents)

    def get_optimal_network(self):
        """
                 Returns the best network over all generation with it's fitness score

                 :param object:

                 :return PeepoNetwork:  The best network over all generations

                 :return float:  The fitness score for the best network

         """
        return self.best_chromosome[1], self.best_chromosome[0]

    def check_convergence(self):
        """
                 Keeps tracks of the performance over the generations.
                 If the actual fitness score is less then the previous one,
                 replace the initial selected parents (see get_parents())
                 by a mix of the actual selected parents and the selected parents of the previous generation.
                 Returns True when the convergence criteria is reached

                 :param object:

                 :return boolean:  True or False

         """
        if len(self.convergence_history) <= self.convergence_period :
            self.convergence_history.append(self.population_fitness_score)
        else:
            self.convergence_history.append(self.population_fitness_score)
            self.convergence_history.pop(0)
            mean = np.mean(self.convergence_history, dtype=np.float64)
            std = np.std(self.convergence_history, dtype=np.float64)
            acceptable_std = mean * self.convergence_sensitivity/2.
            if std < acceptable_std:
                return True

        prev_fitness   = self.last_generation[0]
        actual_fitness = self.population_fitness_score

        if actual_fitness < prev_fitness:
            actual_fitness = (prev_fitness+actual_fitness)/2
            prev_parents = self.copy_parents(self.last_generation[1])
            actual_parents =self.copy_parents(self.selected_parents)
            self.selected_parents = np.concatenate((prev_parents, actual_parents), axis=0)
            random.shuffle(self.selected_parents)
            self.selected_parents = self.copy_parents(self.selected_parents[0:self.number_of_parents])

        self.last_generation[0] =  actual_fitness
        self.last_generation[1] = copy.copy(self.selected_parents)
        return False

    def copy_parents(self, parents):
        a_copy = []
        [a_copy.append([copy.copy(par[0]), copy.copy(par[1])]) for par in parents]
        return a_copy

    def evolve(self, population):
        """
                 Generates the new population array based on the fitness score contained
                 in the returned parameter population

                 :param population: an array of shape (Npop,2) with
                        population[i][0] = fitness score returned from the calling function
                        population[i][1] = the BN-network

                 :param min_fitness_score: a treshold by which the best parents will be selected

                 :return avg_fitness:  a global score for the received population

                 :return population: an array of shape (Npop,2) with
                        population[i][0] = 0
                        population[i][1] = the BN-network with the best parents and their offsprings,

                 :return boolean: True if a certain convergence criteria has been reached otherwise returns False

         """
        self.population = population
        if(len(self.population) == 0):#check if the population has not collapsed
            return -1, self.population, False
        self.get_parents()
        converging = self.check_convergence()
        if converging:
            return self.population_fitness_score, self.population, True
        if self.population_fitness_score < 0:  # check if there is no best population
            return -1, self.population, False
        #-cross-over
        selected_offsprings = self.cross_over()#a member of the offspring array has the shape [0, network, P_mut_top (float), _mut_cpd (float)]
        #-mutate the ofsprings
        random.seed()
        #-check how much chromosomes are left and add parents if necessary
        while True:
            n_chrom = len(self.selected_parents) + len(selected_offsprings)
            if n_chrom >= self.npop:
                break
            [selected_offsprings.append([x[0],x[1],1.,1.]) for x in self.selected_parents]#those parent enter the offsprings pool which can mutate!
        #-mutation
        for s, offspring_ in enumerate(selected_offsprings):
            offspring = copy.copy(offspring_)
            mut_top = offspring_[2]
            mut_cpd = offspring_[3]
            #-check if treshold are reached and mutate accordingly
            if mut_top < self.p_mut_pop:
                network = self.prune_or_grow(offspring[1])
                offspring[1] = network.assemble()
                selected_offsprings[s] = offspring
            if mut_cpd < self.p_mut_cpd:
                network = self.mutate_cpds(offspring[1])
                offspring[1] = network.assemble()
                selected_offsprings[s] = offspring
        #-collecting parents and offsprings
        random.shuffle(selected_offsprings)
        self.population = []
        [self.population.append(par) for par in self.selected_parents]
        [self.population.append([off[0],off[1]]) for off in selected_offsprings]

        #-prune the population to Npop
        self.population  = self.population[0:self.npop]
        return self.population_fitness_score, self.population, False

    def cross_over(self):
        """
                Generates the offsprings array based on the selected_parents

                :param : self object

                :return: an array of size Qx4 with Q (variable) the number of offsprings and
                            selected_offsprings[0] = 0
                            selected_offsprings[1] = the BN-network
                            selected_offsprings[2] = the topology probablity mutation
                            selected_offsprings[3] = the cpd probablity mutation

        """
        selected_offsprings = []
        mating_couples = list(combinations_with_replacement(self.selected_parents,2))
        #filter  out the couples of same chromosomes
        mating_couples= list(filter(lambda x: x[0][1] != x[1][1], mating_couples))
        #to control exponential growth, which can occur in some cases, we limit the number of combinations to Npop
        if len(mating_couples)> self.npop:
            random.shuffle(mating_couples)
            # prune the population
            mating_couples  = mating_couples[0:int(self.npop/2.)]
        for n, chrom in enumerate(mating_couples):
            map_1 = self.get_adjency_map(chrom[0][1].get_edges())
            map_2 = self.get_adjency_map(chrom[1][1].get_edges())
            diff = np.abs(map_1 - map_2)
            sum = np.sum(diff)
            mut_top = random.uniform(0,1)
            mut_cpd = random.uniform(0,1)
            if sum == 0:#-> there is no difference in topology between the parents: only the cpds are swapped
                offspring_1 = copy.copy(chrom[0][1])
                offspring_2 = copy.copy(chrom[1][1])
                offspring_1.cpds = copy.copy(offspring_2.cpds)
                offspring_2.cpds = copy.copy(offspring_1.cpds)
                offspring_1.omega_map = copy.copy(offspring_2.omega_map)
                offspring_2.omega_map = copy.copy(offspring_1.omega_map)
                offspring_1.assemble()
                offspring_2.assemble()
                selected_offsprings.append([0, offspring_1, mut_top, mut_cpd])
                selected_offsprings.append([0, offspring_2, mut_top, mut_cpd])
                continue
            if sum > self.treshold:#the difference between parents is too big. We assume cloning of the  parents (will be mutated!)
                selected_offsprings.append([chrom[0][0],copy.copy(chrom[0][1]),1.,1.])
                selected_offsprings.append([chrom[1][0],copy.copy(chrom[1][1]),1.,1.])
                continue
            ''' we now construct offsprings for all other cases:
                if there are q positions in the two adjency matrices, we will then  have 2^q - 2 offsprings
            '''
            indices = np.argwhere(diff == 1)#this contains the tupples of the matrix position where there is a difference
            combinations = [[1, 0],[0,1]]#this is the possible combinations when we have only 1 difference
            if len(indices) > 1:
                combo = np.full(len(indices),2).tolist()
                combinations = np.transpose(get_index_matrix(combo))
            for comb in combinations:
                map = copy.copy(map_1)
                for pos, indice in enumerate(indices):
                    map[indice[0],indice[1]] = comb[pos]
                #check if some leaf nodes get orphan, if yes we set skip to True and skip this combination
                check_map = np.sum(map, axis = 1)
                skip  = False
                for s in check_map:
                    if s == 0:
                        skip = True
                #check if this combination is equal to one of the parents,if yes we set skip to True and skip this combination
                if np.array_equal(map,map_1) or np.array_equal(map,map_2):
                    skip = True
                if skip:
                    continue
                #for each map(combination) we create an offspring
                edges = self.adjency_to_edges(map)
                a_peepo = copy.copy(self.peepo)
                a_peepo.disassemble()
                a_peepo.edges = edges
                for node in a_peepo.get_nodes():
                    incoming_nodes = a_peepo.get_incoming_edges(node)
                    if len(incoming_nodes) == 0:
                        my_cpd = np.full(a_peepo.cardinality_map[node], 1. / a_peepo.cardinality_map[node])
                        my_omega = []
                    else:
                        my_card_parents = []
                        [my_card_parents.append(a_peepo.cardinality_map[nod]) for nod in incoming_nodes]
                        max_omega = 2 * math.pi * np.prod(my_card_parents)
                        my_omega = np.random.rand(a_peepo.cardinality_map[node]) * max_omega
                        my_cpd = self.ga_child_cpd(my_card_parents, my_omega)
                    a_peepo.add_cpd(node, my_cpd)
                    a_peepo.add_omega(node, my_omega)
                a_peepo.assemble()

                #check whether the combination is a cloning of one parent, if YES mutation will occur
                if np.array_equal(comb[0],comb[1]):
                    mut_top = 1.
                    mut_cpd = 1.
                else:
                    mut_top = random.uniform(0, 1)
                    mut_cpd = random.uniform(0, 1)

                my_chromosome = [0, a_peepo, mut_top,mut_cpd]
                selected_offsprings.append(my_chromosome)
        return selected_offsprings


    def adjency_to_edges(self,map):
        """
                Returns array containing the tuples of the edges from a (pseudo)-adjency list of the topology
                i.e. a len(leaf_nodes)xlen(root_nodes) matrix with
                0 and 1's

                :param map: a matrix of shape (len(leaf_nodes),len(root_nodes)) containing the adjency matrixa

                :return: array containing the tuples of the edges
        """
        edges = []
        for col, root in enumerate(self.root_nodes):
            for row, leaf in enumerate(self.leaf_nodes):
                if map[row][col] == 1:
                    edges.append([root,leaf])
        return edges

    def get_adjency_map(self, edges):
        """
                Returns a (pseudo)-adjency list of the topology
                i.e. a len(leaf_nodes)xlen(root_nodes) matrix with
                0 and 1's

                :param edges: array containing the tuples of the edges

                :return: a matrix of shape (len(leaf_nodes),len(root_nodes)) containing the adjency matrix
        """
        map = np.zeros((len(self.leaf_nodes), len(self.root_nodes)))
        for i, root in enumerate(self.root_nodes):
            for k, leaf in enumerate(self.leaf_nodes):
                for x in edges:
                    if x[1] == leaf and x[0] == root:
                        map[k][i] = 1
        return map



    def prune_or_grow(self,network):
        """
        Used in the framework of Genetic Algorithm to mutate the topology of a given chromosome
        by pruning (if enough incoming edges are present i.e. the maximum possible incoming nodes)
        or by adding a possible parent node not yet incoming in the node choosen.
        Both the node to be pruned/grown and the parent to be rejected/adopted are
        selected randomly.
        This is thus a pure random game approach.
        TO DO : lan node are not yet possible

        :param :network :  PeepoNetwork object containing all the necessary information (node,edges, cpd)
        :return: a fully topology wise, mutated PeepoNetwork object

        """
        nodes = network.get_nodes()
        root_nodes = network.get_root_nodes()
        edges = network.get_edges()
        # lans_nodes = network.get_lan_nodes()
        # leaf_nodes = network.get_leaf_nodes()
        dice = random.randint(len(root_nodes), len(nodes)-1 )
        incoming_edges = network.get_incoming_edges(nodes[dice])
        cpds = network.cpds
        omega_map = network.omega_map
        network.disassemble()
        random.shuffle(incoming_edges)
        new_edges = edges
        if len(incoming_edges) == len(root_nodes):
            new_edges = [x for x in edges if x != (incoming_edges[0], nodes[dice])]
        else:
            candidate_nodes = [x for x in root_nodes if not x in incoming_edges]
            random.shuffle(candidate_nodes)
            new_edges.append((candidate_nodes[0], nodes[dice]))
        network.edges = new_edges
        network.pomegranate_network = None
        incoming_edges = network.get_incoming_edges(nodes[dice])
        # print('incoming edges after mutation : ', incoming_edges)
        parents_card = [self.cardinality_map[x] for x in incoming_edges]
        new_cpd = self.ga_child_cpd(parents_card, omega_map[nodes[dice]])
        cpds[nodes[dice]] = new_cpd
        network.cpds = cpds
        network.omega_map = omega_map
        network.assemble()
        '''
        TO CHECK: the edges order are not ordered anymore -> problem ? or not
        '''
        return network

    def mutate_cpds(self, network):
        leafs = network.get_leaf_nodes()
        omega_map = network.omega_map
        cardinality_map = network.get_cardinality_map()
        epsilon = random.uniform(0.05, 0.75)
        for leaf in leafs:
            incoming_edges = network.get_incoming_edges(leaf)
            parents_card = [cardinality_map[x] for x in incoming_edges]
            my_omega = omega_map[leaf]
            my_omega += (0.5 - np.random.rand(len(my_omega))) * epsilon
            my_cpd = self.ga_child_cpd(parents_card, my_omega)
            network.omega_map[leaf] = my_omega
            network.cpds[leaf] = my_cpd
        return network

    def ga_child_cpd(self,card_parents, omega):
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
        n_comb = np.prod(card_parents)  # type: int
        pdf = []
        for ang in omega:
            pdf_row = []
            for col in range(n_comb):
                pdf_row.append(math.sin(ang * (col + 1) + phase_shift) + 1.2)
            pdf.append(pdf_row)
        return self.normalize_distribution(np.asarray(pdf))

    def normalize_distribution(self, matrix):
        """
        Normalizes the columns of a matrix (i.e. sum matrix[:,i] = 1

        :param matrix: numpy 2 dimensional array
        :return: numpy 2 dimensional normalized array
        """
        factor = np.sum(matrix, axis=0)
        return matrix / factor




