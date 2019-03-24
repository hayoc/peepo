#5/01/2019
import json
import random

from config import ROOT_DIR
from peepo.pp.v3.peepo_network import PeepoNetwork
from peepo.pp.v3.peepo_network import get_topologies
from peepo.tests.parametrized_child_cpd_test import ga_child_cpd  # TO CHANGE


def prune_or_grow(network):
    """
    Used in the framework of Genetic Algorithm to mutate the topology of a given chromosome
    by pruning (if enough incoming edges arz present i.e. the maximum possible incoming nodes)
    or by adding a possible parent node not yet incoming in the node choosen.
    Both the node to be pruned/grown and the parent to be rejected/adopted are
    selected randomly.
    This is thus a pure random game approach.
    TO DO : lan node are not yet possible
    TO DO : how to reconstruct the cpd for the nodes affected
    --> this is simple provided that the omega-parameters can be retrieved from the network


    :param :network :  PeepoNetwork object containing all the necessary information (node,edges, cpd)
    :return: a fully mutated PeepoNetwork object

    example :



        --->

    """
    nodes = network.get_nodes()
    root_nodes = network.get_root_nodes()
    # lans_nodes = network.get_lan_nodes()
    # leaf_nodes = network.get_leaf_nodes()
    print('nodes : ', nodes)
    print('root_nodes : ', root_nodes)
    dice = random.randint(len(root_nodes), len(nodes)-1 )
    print('dice = ', dice, ' and number of nodes : ', len(nodes))
    print('choosen node : ', nodes[dice])
    incoming_edges = network.get_incoming_edges(nodes[dice])


    '''TEMPORARY FOR TESTING PURPOSES'''
    cardinality_map = network.get_cardinality_map()
    print('cardinality_map : ',cardinality_map )
    parents_card = [cardinality_map[x] for x in incoming_edges]
    print('parents card : ', parents_card)
    omega = [0.1,0.2]#TEMPORARY
    old_cpd = ga_child_cpd(parents_card, omega)
    network.add_cpd(nodes[dice], old_cpd)
    '''END OF TEST'''



    # outgoing_edges = network.get_outgoing_edges(nodes[dice])
    edges = network.get_edges()
    print('edges :  ', edges)
    print('incoming edges for ', nodes[dice], ':  ', incoming_edges)
    random.shuffle(incoming_edges)
    print('shuffled_nodes : ', incoming_edges)
    new_edges = edges
    if len(incoming_edges) == len(root_nodes):
        print('removing edge')
        new_edges = [x for x in new_edges if x != (incoming_edges[0], nodes[dice])]
    else:
        candidate_nodes = [x for x in root_nodes if not x in incoming_edges]
        random.shuffle(candidate_nodes)
        new_edges.append((candidate_nodes[0], nodes[dice]))
        print('candidate nodes ', candidate_nodes)
    network.edges = new_edges
    print('new edges : ',network.edges )
    incoming_edges = network.get_incoming_edges(nodes[dice])
    print('new incoming edges : ', incoming_edges)
    cardinality_map = network.get_cardinality_map()
    print('cardinality_map : ',cardinality_map )
    parents_card = [cardinality_map[x] for x in incoming_edges]
    print('parents card : ', parents_card)
    new_cpd = ga_child_cpd(parents_card, omega)
    print('initial cpd for node : ', nodes[dice] ,' = ', network.get_cpds(nodes[dice]))
    network.add_cpd(nodes[dice], new_cpd)
    print('modified cpd for node : ', nodes[dice], ' = ', network.get_cpds(nodes[dice]))
    '''
    TO CHECK: the edges order are not order anymore -> problem ? or not
    '''





if __name__ == '__main__':
    case = 'color_recognition'
    with open(ROOT_DIR + '/resources/' + case + '.json') as json_data:
        json_object = json.load(json_data)
    peepo = PeepoNetwork()
    pp_network = peepo.from_json(json_object)
    topologies = get_topologies(pp_network)
    peepo.edges = topologies[20]['edges']
    prune_or_grow(peepo)

