#versie 19/12/2018
import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from peepo.pp.v3.sensory_input import SensoryInput
from pomegranate import *

from peepo.playground.simple_color_recognition.CeePeeDees import CPD
from peepo.utilities.lattices import Lattices
from peepo.utilities.utilities import Utilities


class SensoryInputVirtualPeepo(SensoryInput):
    def __init__(self, obj):
        super().__init__()
        self.peepo = obj

    def action(self, node, prediction):
        a = 0

    def value(self, name):
        expected_result = self.peepo.expected_result
        cpds = []
        for i in range(0,len(expected_result)):
            cpds.append(['LEN_WORLD_'+str(i), CPD.create_fixed_parent(2, state = int(expected_result[i]))])
        for i, node in enumerate(self.peepo.nodes):
            for j in range(0,len(cpds)):
                if name == cpds[j][0]:
                    return cpds[j][1]

    # def value(self, name):
    #     expected_result = self.peepo.expected_result
    #     cpds = {}
    #     for i in range(0,len(expected_result)):
    #         cpds.update({'WORLD_'+str(i): CPD.create_fixed_parent(2, state = int(expected_result[i]))})
    #     for i, node in enumerate(self.peepo.nodes):
    #         if name == node[0]:
    #             return cpds[name]

class MyClass(object):
    def __init__(self, case):
        self.case = case
        self.results = []
        self.networx_test = nx.DiGraph()
        self.networx_fixed = nx.DiGraph()
        self.pommy_test  = BayesianNetwork()
        self.networx = nx.DiGraph()
        self.pommy = BayesianNetwork()
        self.best_error = math.inf
        self.best_topology = [0,0,nx.DiGraph,0]#[error, entropy, networkx DiGraph, loop]
        self.dictionary = []
        self.header = {}
        self.nodes_0 = []
        self.edges_0 = {}
        self.nodes = []
        self.edges = {}
        self.cpds = {}
        self.colors_dictionary ={}
        self.colors_table =[]
        self.colors_cpd = []
        self.learning_data = {}
        self.nummber_of_colors = 0
        self._util = Utilities(case)
        self._lat = Lattices(self._util)
        self.expected_result = [0, 0, 0]
        self.loop = 0
        self.summary = {}
        self.pom_nodes = {}
        self.summary_test = {}
        self.pom_nodes_test = {}
        self.pixel_states = {'RON_BEN_0':0,'RON_BEN_1':0,'RON_BEN_2':0,'RON_BEN_3':0}
        self.all_pixel_states = {}

    '''.................. vvvvvvvvvvvvvvvvv  TEMPORARY  vvvvvvvvvvvvvvvvvvvvvv ..................................'''

    def sensory_input_value(self, name):
        if name == 'LEN_WORLD_0':
            expected_result = CPD.create_fixed_parent(2, state = self.expected_result[0], modus = 'status')
        if name == 'LEN_WORLD_1':
            expected_result = CPD.create_fixed_parent(2, state = self.expected_result[1], modus = 'status')
        if name == 'LEN_WORLD_2':
            expected_result = CPD.create_fixed_parent(2, state = self.expected_result[2], modus = 'status')
        return expected_result


    def do_simple_inference(self):
        total_prediction_error_size = 0
        for index, node in enumerate(self.predict()):
            node_name = self.pommy_test.states[index].name
            if 'LEN' in node_name:
                # print('index : ', index)
                prediction = np.array([x[1] for x in sorted(node.items(), key=lambda tup: tup[0])])
                observation = self.sensory_input_value(node_name)
                prediction_error_size = self.error_size(prediction, observation)
                total_prediction_error_size += prediction_error_size
        return total_prediction_error_size

    def predict(self):
        """
        Predicts the leaf nodes (i.e. the observational nodes) based on the root nodes (i.e. the belief nodes)
        :return: prediction for all leaf nodes, a prediction is a probability distribution
        :rtype: list of Distributions
        #TODO: A fundamental problem with PP?: Cannot do prediction>error minimization with one loop per node,
        #TODO: since a sister LEN node which does not yet have the correct input will revert the hypothesis update.
        """
        evidence = self.get_root_values()
        #self.pommy_test.bake()
        return self.pommy_test.predict_proba(evidence)

    def error(self,pred, obs):
        """
        Calculates the prediction error as the residual of subtracting the predicted inputs from the observed inputs
        :param pred: predicted sensory inputs
        :param obs: observed sensory inputs
        :return: prediction error
        :type pred : np.array
        :type obs : np.array
        :rtype : np.array
        """
        return obs - pred


    def error_size(self,pred, obs):
        """
        Calculates the size of the prediction error as the Kullback-Leibler divergence. This responds the magnitude
        of the prediction error, how wrong the prediction was.
        :param pred: predicted sensory inputs
        :param obs: observed sensory inputs
        :return: prediction error size
        :type pred : np.array
        :type obs : np.array
        :rtype : float
        """
        scalar = 0.0
        for i in range(0, len(obs)):
            s1 = np.argmax(pred)
            s2 = np.argmax(obs)
            scalar += (s1-s2)*(s1-s2)
        # print('prediction : ', pred, '  observation : ', obs, ' ----> error : ', scalar)
        return scalar#entropy(obs, pred)

    def get_root_values(self):
        return {x.name: x.distribution.mle() for x in self.get_roots()}

    def get_roots(self):
        return [x for x in self.pommy_test.states if 'RON' in x.name]

    def get_leaves(self):
        return [x for x in self.pommy_test.states if self.LEN in x.name]

    def get_node_index(self, node_name):
        for x, state in enumerate(self.pommy_test.states):
            if state.name == node_name:
                return x
        raise ValueError('Node %s does not exist in network.', node_name)


    '''**********************   ^^^^^^^^^^^^^^^ END OF TEMPORARY  ^^^^^^^^^^^^^^^^^^^ ***************************** '''

    def get_my_colors(self):
        evidence = []
        cardinality = []
        for i, node in enumerate(self.nodes):
            if 'BEN' in node[0] or 'MEN' in node[0]:
                evidence.append(node[0])
                cardinality.append(node[1]['cardinality'])
        self.colors_dictionary, self.colors_table, self.colors_cpd = self.color_cpd('LEN_WORLD',3,evidence,cardinality)
        self.number_of_colors = self.colors_table.shape[1]

    def color_cpd(self,var,card_var,evidence,cardinality):
        self.all_pixel_states = {}
        table = CPD.get_index_matrix(cardinality)
        for i in range(0,len(table[1])):
            self.all_pixel_states.update({i: []})
            a_dic ={}
            for j in range(0,len(evidence)):
                a_dic.update({evidence[j]:table[j][i]})
            self.all_pixel_states[i] = a_dic
        # print('*********************************************self.all_pixel_states')
        # print(self.all_pixel_states)
        colors ={}
        hi = 1#0.999
        lo = 1-hi
        C = np.prod(cardinality)
        average = 0
        matrix = np.full((3, C), average)
        if 'BENS_1' in evidence and not 'BENS_2' in evidence and 'BENS_3' in evidence and 'BENS_0' in evidence:
            matrix[0] = [average, lo, hi, average,average, lo, hi, average]
            matrix[1] = [average, lo, lo, average,average, lo, lo, average]
            matrix[2] = [average, hi, lo, average,average, hi, lo, average]
        if 'BENS_1' in evidence and not 'BENS_2' in evidence and 'BENS_3' in evidence and not 'BENS_0' in evidence:
            matrix[0] = [average, lo, hi, average]
            matrix[1] = [average, lo, lo, average]
            matrix[2] = [average, hi, lo, average]
        if 'BENS_1' in evidence and 'BENS_2' in evidence and 'BENS_3' in evidence and not 'BENS_0' in evidence:
            matrix[0] = [lo, lo, lo, lo, hi, lo, hi, lo]
            matrix[1] = [hi, lo, hi, lo, lo, hi, lo, hi]
            matrix[2] = [lo, hi, lo, hi, lo, lo, lo, lo]
        if 'BENS_0' in evidence and 'BENS_1' in evidence and 'BENS_2' in evidence and 'BENS_3' in evidence:
            matrix[0] = [lo, lo, lo, lo, hi, lo, hi, lo, lo, lo, lo, lo, hi, lo, hi, lo]
            matrix[1] = [hi, lo, hi, lo, lo, hi, lo, hi, hi, lo, hi, lo, lo, hi, lo, hi]
            matrix[2] = [lo, hi, lo, hi, lo, lo, lo, lo, lo, hi, lo, hi, lo, lo, lo, lo]
        for i, node in enumerate(evidence):
            colors.update({node:table[i]})
        return colors,table.astype(int), matrix.astype(int)


    # def set_color(self, color):
    #     col = self.colors_table[:, color]
    #     for i in range(0,len(col)):
    #         node = 'BENS_'+ str(i)
    #         self.pgmpy.get_cpds(node).values = CPD.RON_cpd(node, self.pgmpy.get_cardinality(node), mu = int(col[i])).values




    def create_learning_data(self):
        self.get_my_colors()
        self.learning_data = []
        ben_nodes = [x for x in self.nodes if "BEN" in x[0]]
        world_nodes = [x for x in self.nodes if "WORLD" in x[0]]

        for i, node in enumerate(ben_nodes):
            self.learning_data.append(self.colors_table[i].tolist())

        for i, node in enumerate(world_nodes):
            shape = self.colors_cpd.shape
            reshaped_cpd = self.colors_cpd.reshape(shape[0], int(np.prod(shape) / shape[0]))
            for hue in range(0, 3):
                if str(hue) in node[0]:
                    self.learning_data.append(reshaped_cpd[hue, :].tolist())
        self.learning_data = np.asarray(self.learning_data).transpose().astype(int)

    def do_inference(self, models):
        error = 0
        for key in models:
            error += models[key].process()
        return error

    def  get_pommy_root_cpd(self,status):
        if status == 0:
            a_dic = [{0:0.01,1:0.99}]
        if status == 1:
            a_dic = [{0: 0.99, 1: 0.01}]
        return a_dic



    def test_topology(self, entropy):
        self.networx_test = self.networx.copy()
        self.pommy_test, self.pom_nodes_test, self.summary_test = self._util.translate_digraph_to_pomegranate(self.networx_test)
        self.pommy_test.bake()
        #model = {'main': GenerativeModel(SensoryInputVirtualPeepo(self), self.pgmpy_test)}
        self.expected_result = {'LEN_WORLD_0':0,'LEN_WORLD_1':0,'LEN_WORLD_2':0}
        ''' ------ going through all possible "colors'''
        error = 0
        for color in range(0, self.number_of_colors):
            self.pommy_test.thaw()
            pixel_states = self.all_pixel_states[color]
            for m, pixel in enumerate(pixel_states):
                self.pixel_states[pixel] = self.all_pixel_states[color][pixel]
                a_dic = self.get_pommy_root_cpd(self.pixel_states[pixel])
                root_index = self.get_node_index(pixel)
                self.pommy_test.states[root_index].distribution.parameters = a_dic
            self.pommy_test.bake()
            shape = self.colors_cpd.shape
            reshaped_cpd = self.colors_cpd.reshape(shape[0], int(np.prod(shape) / shape[0]))
            self.expected_result = reshaped_cpd[:,int(color)]
            error += self.do_simple_inference()
        error /= (self.number_of_colors*len(self.get_roots()))
        self.results.append([entropy, error])
        if error <= self.best_error:
            self.best_error = error
            self.best_topology[0] = error
            self.best_topology[1] = entropy
            self.best_topology[2] = self.networx_test
            self.best_topology[3] = self.loop
        self.loop += 1


    def add_edges(self, topology):
        self.networx.remove_edges_from(self.edges)
        self.edges = []
        shape = np.asarray(topology).shape
        ''' let's first remove all void nodes  ----> not necssary -----> delete the code ??'''
        nodes_to_remove = []
        #rows = np.sum(topology, axis = 1)
        # for row in range(0, len(rows)):
        #     if rows[row] == 0:
        #         nodes_to_remove.append('WORLD_' + str(row))
        columns = np.sum(topology, axis=0)
        for column in range(0, len(columns)):
            if columns[column] == 0:
                nodes_to_remove.append('RON_BEN_' + str(column))
        self.networx.remove_nodes_from(nodes_to_remove)
        self.nodes = self.networx.nodes(data = True)
        for column in range(0,shape[1]):
            for row in range(0,shape[0]):
                if topology[row][column] == 1:
                    parent = 'RON_BEN_' + str(column)
                    child  = 'LEN_WORLD_'+ str(row)
                    self.networx.add_edge(parent, child)
        self.edges = self.networx.edges()


    def add_dummy_cpds(self):
        for i, node in enumerate(self.nodes):
            cardinality = node[1]['cardinality']
            if ('BEN' in node[0]) or ('MEN' in node[0]):
                self.nodes[i][1]['cpd'] = CPD.create_fixed_parent(cardinality, modus = 'uniform')
            else:
                incoming_nodes = self.networx.in_edges(node[0])
                if len(incoming_nodes) == 0:
                    self.nodes[i][1]['cpd'] = CPD.create_random_child(cardinality, modus = 'orphan')
                    continue
                card_parent = []
                for  m, n in enumerate(incoming_nodes):
                    par = self.networx.node[n[0]]['cardinality']
                    card_parent.append(par)
                self.nodes[i][1]['cpd'] = CPD.create_random_child(cardinality, card_parent)

    def update_network(self):
        ''' update the data associated with the nodes'''
        for i, node in enumerate(self.nodes):
            self.nodes[i][1]['childs'] = []
            self.nodes[i][1]['parents'] = []
            self.nodes[i][1]['parents_cardinality'] = []
            for k, edge in enumerate(self.networx.edges()):
                if node[0] == edge[0]:
                    self.nodes[i][1]['childs'].append(edge[1])
            cardinality = node[1]['cardinality']
            if ('BEN' in node[0]) or ('MEM' in node[0]):
                self.nodes[i][1]['parents'] = []
            else:
                incoming_nodes = self.networx.in_edges(node[0])
                if len(incoming_nodes) == 0:
                    self.nodes[i][1]['parents']  = []
                    continue
                card_parent = []
                paren = []
                for  m, n in enumerate(incoming_nodes):
                    if n[1] == node[0]:
                        par = self.networx.node[n[0]]['cardinality']
                        paren.append(n[0])
                        card_parent.append(par)
                    self.nodes[i][1]['parents_cardinality'] = card_parent

                    self.nodes[i][1]['parents'] = paren
        for node, out_degree in self.networx.out_degree_iter():
            if out_degree == 0:
                self.networx.node[node]['childs'] = []

        self.nodes = self.networx.nodes(data=True)



    def do_it(self):
        '''EXPLANATIONS'''
        self.networx_fixed , self.summary, self.dictionary, self.header = self._util.get_network()
        self.networx = self.networx_fixed.copy()
        self.networx_test= self.networx_fixed.copy()
        self.nodes = self.networx.nodes(data=True)
        print('Dictionary : ', self.dictionary)

        ''' -------------- Constructing all possible topologies, 
                              --> option : restrain the number with the treshold : 
                                        0 -> all possible topologies, 100 -> only the fully connnected topology'''
        possible_topologies  = self._lat.get_possible_topologies(treshold = 50)#setting the entropy at a 50% -> only topologies with an entropy >= 0.5 will be considered
        print("Possible topologies : ", len(possible_topologies))
        entropy = 0
        count = 0#TEMPORARY
        ''' -------------- walking through all toplogies'''
        for topology in possible_topologies:
            entropy = topology[1]
            if entropy == 0:
                continue#safeguard
            print('Loop *-> ', self.loop + 1, ' of ', len(possible_topologies))
            topo  = topology[0]
            self.networx = self.networx_fixed.copy()
            ''' ----------- for each topology we construct the edges and update dummy cpd (necessary as the shape of the LENs cpd's can change
                            depending on the number of incoming nodes'''
            self.add_edges(topo)
            self.add_dummy_cpds()
            ''' update the data associated with the nodes'''
            self.update_network()
            self.create_learning_data()
            ''' ----------- convert DiGraph topomegrante'''
            self.pomi, self.pom_nodes, self.summary = self._util.translate_digraph_to_pomegranate(self.networx)
            self.pomi.bake()
            '''------------ askpomegranate to guess the best cpd's of the LANs and LENs 
                             -> provide pomegranate with the learning data'''
            self.pomi.fit(self.learning_data)
            self.update_network()

            '''-------------- Testing the constructed topology'''
            self.test_topology(entropy)

            '''following  4 lines to remove : just use to check whether the algorithms are correct regarding the edges building'''
            count += 1
            #print('edges : ', self.edges)
            #
            # if count > 400:
            #     break
        print('Check -> number of processed topologies in loop : ', count)
        # print('My colors : ')
        # print(self.colors_table)
        # print(self.colors_cpd)
        '''TO DO ----------------------------------------------------
                a) add random cpds , convert to pgmpy BN, 
                b) enbedd the skeleton loop  within the learning loop->
                    loop through all possible colors and the expected classification
                    -- > for each skeleton with the possible color as BEN, make  pgmpy guess the best cpd's 
                         with the method class 
                                   in pgmpy.estimators.BayesianEstimator.BayesianEstimator(model, data, **kwargs)[source]
                                            estimate_cpd(node, prior_type='BDeu', pseudo_counts=[], equivalent_sample_size=5)[source]
                    -- > make inference and calulate the 'error (to be determined)
                    ---> log the error as a tuple (error, 'entropy of the skeleton')
                c) create output (grapgh?)
                    
            
            
            '''



        '''  the methods have to be completed to cope with a general case i.e. BENS,MEMS,LANS, MOTORs, WORLDs
        but for the moment being we just assume there are only BEN's and WORLD's'''

        # self.networx.add_edge('BENS_1','WORLD_1')
        # self.networx.node['BENS_1']['cpd'] = [0.8,0.2]
        # self.networx.node['WORLD_2']['cpd'] = [[0.8, 0.2, 0.5,0.3],[0.2,0.8,0.5,0.7]]
        ''' if a best model has ben found, save it -> first update the Utility class object and save it'''
        # self._util.update_networkx(self.networx, self.dictionary, self.header)
        # self._util.save_network()
        # self._util.update_pgmpy(self.pgmpy, self.dictionary, self.header)
        # self._util.save_pgmpy_network()
        self.draw()
        self.draw_xy()
        return self.results

    def draw_xy(self):
        x = []
        y = []
        s = []
        color = []
        best_x = 0
        best_y = 0
        for i in range(0, len(self.results)):
            x.append(self.results[i][0])
            y.append(self.results[i][1])
            if i == self.best_topology[3]:
                best_x = self.results[i][0]
                best_y = self.results[i][1]
                s.append(60)
                color.append("r")
            else:
                s.append(20)
                color.append("b")
        plt.scatter(x, y, s = s, c=color, alpha=0.5)
        plt.xlabel("Complexity of topology")
        plt.ylabel("Average error over all colors")
        plt.show()



    def draw(self):
        '''TO REMOVE LATER'''
        plt.figure(figsize=(10, 5))
        pos = nx.circular_layout(self.best_topology[2], scale=2)
        #node_labels = nx.get_node_attributes(self.networx, 'cpd')
        nx.draw(self.best_topology[2], pos, node_size=1200, node_color='lightblue',
                linewidths=0.25,  font_size=10, font_weight='bold', with_labels=True)
        plt.text(1,1, 'Topology nr. : ' +str(self.best_topology[3]))
        plt.show()

def main():
    case = 'simple_color_recognition'
    mycase = MyClass(case)
    results = mycase.do_it()
    #print(results)

####################################################################################
############################### BEGIN HERE #########################################
####################################################################################

if __name__ == "__main__":
    # logging.basicConfig()
    # logging.getLogger().setLevel(logging.DEBUG)
    main()