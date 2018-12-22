import os
import datetime
import json
import networkx as nx
import numpy as np
import copy
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
import re
import itertools
from peepo.utilities.lattices import  Lattices
import matplotlib.pyplot as plt
#from peepo.visualize.graph import draw_network
#import matplotlib.pyplot as plt
from pomegranate import *
from peepo.utilities.CPD_Pomgranate import CPD_P


class Utilities(object):
    def __init__(self, file):
        ''' no object creation -> opportune  ?'''
        self.keywords = ['BENS','MEMS','LANS','MOTOR','WORLD']
        self.standard_nodes = {'RONS': {'BENS': [], 'MEMS': []}, 'LANS': {'LANS': []},
                               'LENS': {'MOTOR': [], 'WORLD': []}}
        self.file = file
        self.get_json_path(file)
        self.pgmpy_object = BayesianModel()
        self.networkx_object = nx.DiGraph()
        self.pomegranate_object = BayesianNetwork()
        self.header =''
        self.dictionary =[]
        self.summary = {'raw_cpd':{},'pom_cpd':{},'cardinality':{},'edges':{},'parents':{},'parents_cardinalities':{},'childs':{}}
        self.summary_0 = {'raw_cpd': {}, 'pom_cpd': {}, 'cardinality': {}, 'edges': {}, 'parents': {},
                        'parents_cardinalities': {}, 'childs': {}}

    def get_nodes_in_family(self, family, attributes=False):
        nw_nodes = self.networkx_object.nodes()
        nw_dim = np.asarray(nw_nodes).ndim
        nodes = []
        for i, node in enumerate(nw_nodes):
            if nw_dim > 1:
                node = node[0]
            if family in node:
                nodes.append(node)
        return nodes

    def check_json_path(directory):
        """
        Checks whether the necessary project_repository directory exists.
        If not, creates it

        :param directory: the mother directory to search from downwards

        :type directory: string
        :rtype : none
        """
        if not os.path.exists(directory + '\project_repository\\'):
            os.makedirs(directory + '\project_repository\\')

    def get_json_path(self, file):
        """
        Creates a string containing the full path for the filename passed
        so it will be saved in the project_repository directory

        :param filename: filename without path or extension
        :return: a full path for the file

        :type filename :string
        :rtype : string
        """
        levels = 5
        common = os.path.dirname(os.path.realpath(__file__))
        for i in range(levels + 1):
            common = os.path.dirname(common)
            if 'peepo\peepo' not in common:
                break
        Utilities.check_json_path(common)
        self.file = str(common + '\project_repository\\'+ file +'.json')
        print('in get_json_path :' , self.file)

    def save_json(self, astring):
        """
        This helping function is only needed to have the json file  formatted in a user friendly way
        as the "dump" method does not provide a lot of possibilities to get it "pretty"

        :param file :the ull path of the json file
        :param astring: the name of the string containing the whole information
        :return: void
        :type file: string
        :type astring : string
        :rtype : void
        """
        text_file = open(str(self.file), "w")
        '''remove all LF written by the dump method'''
        astring = re.sub('\n', '', astring)
        '''For keywords -> insert LF and tabs'''
        astring = re.sub('\"Identification', '\n\"Identification', astring)
        astring = re.sub('\"Date', '\n\"Date', astring)
        astring = re.sub('\"Description', '\n\"Description', astring)
        astring = re.sub('\"Train_from', '\n\"Train_from', astring)
        astring = re.sub('\"Frozen', '\n\"Frozen', astring)
        astring = re.sub('\"Nodes', '\n\n\"Nodes', astring)
        astring = re.sub('\"RONS', '\n\t\t\"RONS', astring)
        astring = re.sub('\"BENS', '\n\t\t\t\"BENS', astring)
        astring = re.sub('\"MEMS', '\n\t\t\t\"MEMS', astring)
        astring = re.sub('\"LANS', '\n\t\t\"LANS', astring)
        astring = re.sub('\"LENS', '\n\t\t\"LENS', astring)
        astring = re.sub('\"MOTOR', '\n\t\t\t\"MOTOR', astring)
        astring = re.sub('\"WORLD', '\n\t\t\t\"WORLD', astring)
        astring = re.sub('\"Edges', '\n\n\"Edges', astring)
        astring = re.sub('\"CPDs', '\n\n\"CPDs', astring)
        astring = re.sub('{', '\n\t\t{', astring)
        text_file.write(astring)
        text_file.write('\n')
        text_file.close()

    def translation(self, astring, from_man_to_machine):
        """
        Given an array of tuples (a,b) in dictionary, returns the second element of the tuple where astring was found
        Is used to not loose the users node names as peepo generates standardized names for the corresponding node

        :param dictionary:an array of tuples -> is created in the method : get_network(file)
        :param astring: the name of the node passsed by the user
        :param from_man_to_machine: an integer -> 0 when we want the translation for the user give name to the standardized name, 1 the other way around
        :return: the corresponding standardized node name
        :type dictionary: np.array
        :type astring : string
        :rtype : string
        """
        source = 0
        target = 1
        if from_man_to_machine == 1:
            source = 1
            target = 0

        for index, item in enumerate(self.dictionary):
            if item[source] == astring :
                break
        return  item[target]

    def clean_edge_list(self, edge_array, parent):
        '''the get functions for the edges, both in networx as pgmpy contain the parent name
            this function removes it from the list'''
        cleaned_list = []
        for a in edge_array:
            if a!= parent:
                cleaned_list.append(a)
        return cleaned_list

    def clean_parent_list(self, parent_array, child):
        '''the get functions for the edges, both in networx as pgmpy contain the parent name
            this function removes it from the list'''
        cleaned_list = []
        for i,a in enumerate(parent_array):
            if a[0]!= child:
                cleaned_list.append(a[0])
        return cleaned_list

    def get_edges(self):
        """
        Creates a dictionary with a node as a key and an array with its child as value
        (the methods get_child give generally a list of tuples (parent,child)

        :param  pgmpy_object: the pgmpy network
        :return: a dictionary with the edges of all the node

        :type fpgmpy_object:adress
        :rtype :dictionary
                """
        edg = self.pgmpy_object.edges()
        edges = dict()
        [edges[str(t[0])].append(str(t[1])) if t[0] in list(edges.keys()) else edges.update({str(t[0]): [str(t[1])]}) for t in edg]
        return edges


    def get_nodes_and_attributes(self):
        """
        Creates an  array  of tuple with a node as element 0 and a dictionary with cardinalities and cpd as key's and
         the key cardinality returns an int
         the key cpd a 2 dimensional matrix

        :param  pgmpy_object: the pgmpy network
        :return: array  of tuple with a node as element 0 and a dictionary with cardinalities and cpd as key's

        :type  :pgmpy_object:adress
        :rtype :array of tuples
        """
        nodes = self.pgmpy_object.nodes()
        nod_and_attributes = []
        [nod_and_attributes.append((str(node),{'cardinality':int(self.pgmpy_object.get_cardinality(node)), 'cpd': self.pgmpy_object.get_cpds(node).values.astype(float)}))  for i,node in enumerate(nodes)]
        #need to reshape the cpds when more than 1 parent
        for i,node in enumerate(nod_and_attributes):
            shape = nod_and_attributes[i][1]['cpd'].shape
            dimension = nod_and_attributes[i][1]['cpd'].ndim
            if dimension  > 2:
                col = int(np.prod(shape)/shape[0])
                nod_and_attributes[i][1]['cpd'] = nod_and_attributes[i][1]['cpd'].reshape(shape[0], col)
            nod_and_attributes[i][1]['cpd'] = nod_and_attributes[i][1]['cpd'].tolist()
        return nod_and_attributes

    def translate_pgmpy_to_digraph(self):
        """
        Converts a pgmpy network into a networkx network

        :param  pgmpy_object: the pgmpy network
        :return networkx : networkx network

        :type  :pgmpy_object:adress
        :rtype :networkx:adress
        """
        self.networkx_object = nx.DiGraph()
        edges = self.pgmpy_object.edges()
        nodes_and_attributes =  self.get_nodes_and_attributes()
        self.networkx_object.add_nodes_from(nodes_and_attributes)
        self.networkx_object.add_edges_from(edges)
        return

    def translate_digraph_to_pgmpy(self, digraf):
        """
        Converts a pgmpy network into a networkx network

        :param  pgmpy_object: the pgmpy network
        :return networkx : networkx network

        :type  :pgmpy_object:adress
        :rtype :networkx:adress
        """
        self.pgmpy_object, x,y = self.get_pgmpy_network(from_object = True, digraph = digraf )
        return self.pgmpy_object


    def translate_digraph_to_pomegranate(self, digraf):
        """
        Converts a pgmpy network into a networkx network

        :param  pgmpy_object: the pgmpy network
        :return networkx : networkx network

        :type  :pgmpy_object:adress
        :rtype :networkx:adress
        """
        self.pomegranate , self.summary, x,y = self.get_pomegranate_network(from_object = True, digraph = digraf )
        return self.pomegranate_object, self.summary
    

    def update_networkx(self, networkx, dic, header):
        self.header = header
        self.dictionary = dic
        self.networkx_object = networkx

    def update_pgmpy(self, pgmpy, dic, header):
        self.header = header
        self.dictionary = dic
        self.pgmpy_object = pgmpy

    def save_pgmpy_network(self):
        """
                Saves the passed pgmpy_object class object in a json file
        """
        self.translate_pgmpy_to_digraph()
        self.save_network()
        return


    def save_network(self):
        """
        Saves the passed networkx class object in a json file

        """
        data = self.get_empty_canvas()
        data["header"] = self.header
        nw_nodes = self.networkx_object.nodes(data = True)
        nw_edges = self.networkx_object.edges()
        keywords = self.keywords
        nodes = copy.deepcopy(self.standard_nodes)#{'RONS': {'BENS': [], 'MEMS': []}, 'LANS': {'LANS': []}, 'LENS': {'MOTOR': [], 'WORLD': []}}
        edges = []
        cpds = []
        '''adding edges'''
        for i, node in enumerate(nw_nodes):
            node_name = node[0]
            childs =[]
            for k, edge in enumerate(nw_edges):
                if edge[0] == node_name:
                    childs.append(self.translation(edge[1],1))
            if len(childs) != 0:
                edges.append({self.translation(node_name,1):childs})

        for i, node in enumerate(nw_nodes):
            node_name = node[0]
            cardinality = node[1]['cardinality']
            cpd = node[1]['cpd']
            for pseudonym  in keywords:
                if pseudonym in node_name:
                    node_name_ = self.translation(node_name,1)
                    if pseudonym == 'BENS' or pseudonym == 'MEMS':
                        nodes['RONS'][pseudonym].append([node_name_, cardinality])
                    if pseudonym == 'LANS':
                        nodes['LANS'][pseudonym].append([node_name_, cardinality])
                    if pseudonym == 'MOTOR' or pseudonym == 'WORLD':
                        nodes['LENS'][pseudonym].append([node_name_, cardinality])
            cpds.append({self.translation(node_name,1):cpd})
        data['Nodes'] = nodes
        data['Edges'] = edges
        data['CPDs']  = cpds
        data['header']['Date'] = datetime.datetime.now().strftime("%c")
        self.save_json(json.dumps(data))
        return


    def translate_cpd_to_pomegranate(self, type, node_name,cardinality, cpd, cpd_parents, parents_cardinality, child):
        '''    create ad hoc dictionary or table       '''
        if type == 'RONS':
            dictionary = {}
            for state in range(0, cardinality):
                state_label = str(state)
                dictionary.update({state_label:cpd[state]})
            return DiscreteDistribution(dictionary)
        if type == 'LANS' or type == 'MOTOR' or type == 'WORLD' or type == 'LENS':
            table = CPD_P.get_index_matrix(parents_cardinality)
            shape = table.shape
            cpd_table = []
            cpd_shape = cpd.shape
            for column  in range(0,shape[1]):
                for child_state in range(0, cpd_shape[0]):
                    an_entry = []
                    state_child = str(int(child_state))
                    phi_child = cpd[child_state][column]
                    for parent  in range(0,shape[0]):
                        an_entry.append(str(int(table[parent][column])))
                    an_entry.append(state_child)
                    an_entry.append(phi_child)
                    cpd_table.append(an_entry)
            return ConditionalProbabilityTable(cpd_table,cpd_parents)

    def get_pomegranate_network(self, from_object = False, digraph = None):
        """
        Reads the passed json file and translates it's content to the passed pgmpy class object
        - uses the get_network(file) to read the json file in a networkx format and translate this to pgmpy
        - Creates a dictionary for the nodes in the form of an array of tuples : [(names defines by user, standard name)]

        :param file: : filename without path or extension
        :pgmp_object : the pgmpy object which will be completed
        :return: a dictionary as an array of tuples and the header of the json file

        :type file : string
        :type pgmp_object : pgmpy class object
        :rtype : array of tuples, dictionary

        CAUTION : the method does not perform a check() on the constructed DAG ! -> has to be done in the calling module
        """
        self.pomegranate_object = BayesianModel()
        self.summary = copy.deepcopy(self.summary_0)
        if not (from_object):
            network, dictionary, header =  self.get_network()
        else:
            network = digraph
        nw_nodes = network.nodes(data = True)
        nw_edges = network.edges()
        '''make a pomegranate compatible list of cpd's  and states'''
        for i, node in enumerate(nw_nodes):
            node_name = node[0]
            self.summary['cardinality'].update({node_name:node[1]['cardinality']})
            self.summary['parents'].update({node_name:node[1]['parents']})
            self.summary['parents_cardinalities'].update({node_name: node[1]['parents_cardinality']})
            self.summary['raw_cpd'].update({node_name:node[1]['cpd']})
            self.summary['childs'].update({node_name:node[1]['childs']})

        ''' first the RONs'''
        for i, node in enumerate(nw_nodes):
            node_name = node[0]
            cardinality = self.summary['cardinality'][node_name]
            parent = self.summary['parents'][node_name]
            child = self.summary['childs'][node_name]
            parent_cardinality = self.summary['parents_cardinalities'][node_name]
            cpd = self.summary['raw_cpd'][node_name]
            if len(parent) != 0 :
                continue
            cpd_parents = []
            cpd_p = self.translate_cpd_to_pomegranate('RONS',node_name,cardinality, cpd, cpd_parents, parent_cardinality, child)
            self.summary['pom_cpd'].update({node_name:cpd_p})

        '''2nd  the LANs'''
        for i, node in enumerate(nw_nodes):
            node_name = node[0]
            cardinality = self.summary['cardinality'][node_name]
            parent = self.summary['parents'][node_name]
            child = self.summary['childs'][node_name]
            parent_cardinality = self.summary['parents_cardinalities'][node_name]
            cpd = self.summary['raw_cpd'][node_name]
            if len(parent) == 0 or len(child ) == 0:
                continue
            cpd_parents = []
            for k, par in enumerate(parent):
                cpd_parents.append(self.summary['pom_cpd'][par])
            cpd_p = self.translate_cpd_to_pomegranate('LANS',node_name,cardinality, cpd, cpd_parents, parent_cardinality, child)
            self.summary['pom_cpd'].update({node_name:cpd_p})

        '''3rd the LEAFs'''
        for i, node in enumerate(nw_nodes):
            node_name = node[0]
            cardinality = self.summary['cardinality'][node_name]
            parent = self.summary['parents'][node_name]
            child = self.summary['childs'][node_name]
            parent_cardinality = self.summary['parents_cardinalities'][node_name]
            cpd = self.summary['raw_cpd'][node_name]
            if len(parent) == 0 or len(child) != 0:
                continue
            cpd_parents = []
            for k, par in enumerate(parent):
                cpd_parents.append(self.summary['pom_cpd'][par])
            cpd_p = self.translate_cpd_to_pomegranate('LENS',node_name,cardinality, cpd, cpd_parents, parent_cardinality, child)
            self.summary['pom_cpd'].update({node_name:cpd_p})

        self.pomegranate_object = BayesianNetwork()

        '''adding nodes '''
        pom_edges = []
        pom_nodes = []
        s = 0
        for i, node in enumerate(nw_nodes):
            node_name = node[0]
            s = copy.deepcopy(Node(self.summary['pom_cpd'][node_name], name = node_name))
            pom_nodes.append([node_name,s])
            self.pomegranate_object.add_state(s)

        '''adding edges '''
        for i, edge in enumerate(nw_edges):
            s_parent  = ''
            s_child = ''
            for k, pom in enumerate(pom_nodes):
                if edge[0]== pom[0]:
                    s_parent = pom[1]
                if edge[1] == pom[0]:
                    s_child = pom[1]
            self.pomegranate_object.add_edge(s_parent, s_child)

        return self.pomegranate_object, self.summary, self.dictionary, self.header

    def get_pgmpy_network(self, from_object = False, digraph = None):
        """
        Reads the passed json file and translates it's content to the passed pgmpy class object
        - uses the get_network(file) to read the json file in a networkx format and translate this to pgmpy
        - Creates a dictionary for the nodes in the form of an array of tuples : [(names defines by user, standard name)]

        :param file: : filename without path or extension
        :pgmp_object : the pgmpy object which will be completed
        :return: a dictionary as an array of tuples and the header of the json file

        :type file : string
        :type pgmp_object : pgmpy class object
        :rtype : array of tuples, dictionary

        CAUTION : the method does not perform a check() on the constructed DAG ! -> has to be done in the calling module
        """
        self.pgmpy_object = BayesianModel()
        if not (from_object):
            network, dictionary, header =  self.get_network()
        else:
            network = digraph
        nw_nodes = network.nodes(data = True)
        nw_edges = network.edges()
        '''adding nodes and edges'''
        for i, node in enumerate(nw_nodes):
            node_name = node[0]
            self.pgmpy_object.add_node(node_name)
            for k, edge in enumerate(nw_edges):
                if edge[0] == node_name:
                    self.pgmpy_object.add_edge(node_name, edge[1])
        '''add  cpd's'''
        for i, node in enumerate(nw_nodes):
            parent_nodes = network.in_edges(node[0])
            parent_nodes = self.clean_parent_list(parent_nodes,node[0])
            cpd = node[1]['cpd']
            ''' find the cardinality of the node '''
            cardinality_node = node[1]['cardinality']
            """  cardinality card of parents has to be determined"""
            cardinality_parents = []
            for i,nod in enumerate(parent_nodes):
                cardinality_parents.append(network.node[nod]['cardinality'])
            ''' Depending on the place in the BN and/or the number of parents  the PGMPY CPD methods have another call'''
            if len(cardinality_parents) == 0:
                self.pgmpy_object.add_cpds(TabularCPD(variable=node[0], variable_card= cardinality_node, values=[cpd]))
                continue
            table = TabularCPD(variable=node[0], variable_card= cardinality_node, values=cpd, \
                              evidence=parent_nodes,\
                              evidence_card=np.asarray(cardinality_parents))
            self.pgmpy_object.add_cpds(table)
        '''------TO DELETE-------------'''
        # pgmpy_object.check_model()
        # draw_network(pgmpy_object)
        '''-----------------------------'''
        return self.pgmpy_object,self.dictionary, self.header


    def get_network(self):
        """
        Reads the passed json file and translate it's content in a networkx class object
        - The nodes in the object are renamed so they have a standardized signature
        - Creates a dictionary for the nodes in the form of an array of tuples : [(names defines by user, standard name)]

        :param file: : filename without path or extension
        :return: a networkx class object, dictionary as an array of tuples and the header of the json file

        :type file : string
        :rtype : networkx class object, array of tuples, dictionary
        """
        self.dictionary = []
        self.networkx_object = nx.DiGraph()
        with open(self.file) as f:
            data = f.read()
        '''Remove possible non informative characters'''
        data = re.sub('\n', '', data)
        data = re.sub('\t', '', data)
        data = json.loads(data)
        self.header = data['header']
        '''Feeding G with the nodes'''
        cardinality = {}
        parents = {}
        childs = {}
        cardinality_parents = {}
        for key in data['Nodes'].keys():
            for  secondkey in data['Nodes'][key].keys():
                for c, n  in enumerate(data['Nodes'][key][secondkey]):
                    node = secondkey + "_" + str(c)
                    self.networkx_object.add_node(node, {'cardinality':n[1], 'cpd':[], 'parents':[],'parents_cardinality':[], 'childs':[]})
                    self.dictionary.append((n[0],node))
                    cardinality.update({node:n[1]})#this contains the cardinality of each node with the node name as dictionary entry
                    parents.update({node:[]})
                    cardinality_parents.update({node:[]})
                    childs.update({node:[]})

        '''Feeding G with the edges'''
        edges = []

        for j, pair in enumerate(data['Edges']):
            for parent in pair.keys():
                for child in data['Edges'][j][parent]:
                    parent_ = self.translation(parent, 0)
                    child_  = self.translation(child, 0)
                    edges.append((parent_,child_))
                    parents[child_].append(parent_)
                    childs[parent_].append(child_)
                    cardinality_parents[child_].append(self.networkx_object.node[parent_]['cardinality'])
        np.ravel(edges)
        self.networkx_object.add_edges_from(edges)

        '''gather info about the parent and or childs  of a node'''
        for i, node in enumerate(self.networkx_object.nodes()):
            self.networkx_object.node[node]['parents'] = parents[node]
            self.networkx_object.node[node]['parents_cardinality'] = cardinality_parents[node]
            self.networkx_object.node[node]['childs'] = childs[node]

        '''Feeding G with the  CPD's as nodes attributes'''
        for j, node in enumerate(data['CPDs']):
            for parent, cpd in node.items():
                node_ = self.translation( parent, 0)
                self.networkx_object.node[node_]['cpd'] = cpd
        return self.networkx_object, self.summary, self.dictionary, self.header


    def create_json_file(self, **kwargs):

        """
        EWAMPLE :

        A helping method if the user prefers to create the BN within the code

        :param case_name: the file name without path or extension where the json file will be saved
        :param : **kwargs takes the following variables:
                                                            description = kwargs.get('description', '')
                                                            train_from = kwargs.get('train_from', '')
                                                            cpds = kwargs.get('CPDs', [])
                                                            bens = kwargs.get('BENS',[])
                                                            mems = kwargs.get('MEMS', [])
                                                            lans = kwargs.get('LANS', [])
                                                            motors = kwargs.get('MOTORS', [])
                                                            world = kwargs.get('WORLD', [])
                                                            edges = kwargs.get('Edges', [])
                                                            frozen = kwargs.get('frozen',False)
        .
        .
        .
        :return: void

        :type case_name : string
        :type  :
        .
        .
        .
        :rtype : void
        """
        description = kwargs.get('description', '')
        train_from = kwargs.get('train_from', '')
        cpds = kwargs.get('CPDs', [])
        bens = kwargs.get('BENS',[])
        mems = kwargs.get('MEMS', [])
        lans = kwargs.get('LANS', [])
        motors = kwargs.get('MOTORS', [])
        world = kwargs.get('WORLD', [])
        edges = kwargs.get('Edges', [])
        frozen = kwargs.get('frozen',False)

        #json_tab_file_write = JSONTabIndentFileWriter( Case_name,5a)
        data = self.get_empty_canvas()

        '''       - the 3 next items are for tracking purpose only, not fundamentally necessary'''
        data["header"]['Identification'] = self.file
        data["header"]['Date'] = datetime.datetime.now().strftime("%c")
        data["header"]['Description'] = description
        '''       - the next item gives a file containing possible training data (OPTIONAL)'''
        data["header"]['Train_from'] = train_from

        '''      Frozen tells whether or not the model can be considered as final i.e. is there still "training" needed'''
        data["header"]['Frozen'] = frozen

        '''       - the 5 next lines tells how much nodes  and their names + cardinality the model will start with
                    the names can be any valid python string'''
        bens = [['pooping',2], ['peeing',2], ['constipated',2]]
        mems = [['havenotoiletpaper',2]]
        lans = [['diarhea',2], ['happypoop',2]]
        motors = [['asshole1',2], ['asshole2',2]]
        world = [['toilet1',2], ['toilet2',2], ['garden1',2], ['garden2',2],['doctor',2]]

        '''     - the next items describe the edges as a dictionary
                 -> the dictionary entry is always one of the rootnodes, the array following can only contain LANs or LENs'''
        edges = []

        '''       !! in case we start from scratch and we rely on peepo to find the best BN -> leave this array empty'''
        edges.append({'pooping': ['toilet1', 'diarhea','happypoop']})
        edges.append({'peeing': ['toilet2', 'garden1', 'garden2']})
        edges.append({'constipated': ['doctor']})
        edges.append({'havenotoiletpaper': ['garden1', 'garden2']})
        edges.append({'diarhea': ['toilet1', 'doctor','asshole1', 'asshole2']})
        edges.append({'happypoop': ['garden1', 'garden2','asshole1', 'asshole2']})



        '''       - the next items describe the CPD's  as a dictionary
                  -> the dictionary entry is the corresponding node'''
        cpds = []
        cpds.append({'pooping': [0.5,0.5]})
        cpds.append({'peeing': [0.2,0.8]})
        cpds.append({'constipated': [0.9,0.1]})
        cpds.append({'havenotoiletpaper': [0.6,0.4]})
        cpds.append({'happypoop': [[0.3,0.8],[0.7,0.2]]})
        cpds.append({'diarhea': [[0.8,0.3],[0.2,0.7]]})
        cpds.append({'toilet1': [[0.3,0.8,0.8,0.7],[0.7,0.2,0.2,0.3]]})
        cpds.append({'asshole1': [[0.3,0.8,0.8,0.7],[0.7,0.2,0.2,0.3]]})
        cpds.append({'asshole2': [[0.3,0.8,0.8,0.7],[0.7,0.2,0.2,0.3]]})
        cpds.append({'toilet2': [[0.5, 0.5],[0.5, 0.5]]})
        cpds.append({'doctor': [[0.3,0.8,0.8,0.7],[0.7,0.2,0.2,0.3]]})
        cpds.append({'garden1': [[0.3,0.8,0.8,0.7, 0.8,0.2,0.5,0.5],[0.7,0.2,0.2,0.3,0.2,0.8,0.5,0.5]]})
        cpds.append({'garden2': [[0.3,0.8,0.8,0.7, 0.8,0.2,0.5,0.5],[0.7,0.2,0.2,0.3,0.2,0.8,0.5,0.5]]})

        '''       - feeding the data'''
        data["Nodes"]['RONS']['BENS']  = bens
        data["Nodes"]['RONS']['MEMS']  = mems
        data["Nodes"]['LANS']['LANS']= lans
        data["Nodes"]['LENS']['MOTOR'] = motors
        data["Nodes"]['LENS']['WORLD'] = world
        data["Edges"] = edges
        data["CPDs"] = cpds

        ''' dumping to CASENAME file in jason format'''
        self.save_json(json.dumps(data))

        print("Json file for  - ", self.file, "  - created")

    def create_json_template(self):
        """
        A helping method if the  jason template in the project_repository ditectory has been deleted or corrupted

        :param : void
        :return: void

        :type : void
        :rtype : void
        """
        self.get_json_path("Template")  # creates the right path in which case_name will be saved
        data =  self.get_empty_canvas()
        data['header']['Identification'] = self.file
        '''Filling some dummies to facilitate the user'''
        a_node = ['*',0]
        an_edge = {'*':['&','&','&']}
        a_cpd = {'*':[[0,0,0],[0,0,0]]}
        nodes = []
        edges = []
        cpds = []
        for i in range(0,3):
            nodes.append(a_node)
            edges.append(an_edge)
            cpds.append(a_cpd)

        data['Nodes']['RONS']['BENS'] = nodes
        data['Nodes']['RONS']['MEMS']= nodes
        data['Nodes']['LANS']['LANS'] = nodes
        data['Nodes']['LENS']['MOTOR'] = nodes
        data['Nodes']['LENS']['WORLD'] =nodes
        data['Edges'] = edges
        data['CPDs'] = cpds

        ''' dumping to CASENAME file in jason format'''
        # with open(case_name, 'w') as f:
        #     json.dump(data, f, separators = (",",":"))
        self.save_json(json.dumps(data))
        print("Empty template created")

    def get_empty_canvas(self):
        """
         This method creates a json canvas which will be used for the several json creating method

         :param : void
         :return: a dictionary with the structure of the json file
         :type : non
         :rtype : dictionary
         """

        data = {'header': {'Identification': '', 'Date': '', 'Description': '', 'Frozen': '', 'Train_from': ''},
                'Nodes': {}, 'Edges': [], 'CPDs': []}

        '''       - the 5 next lines tells how much nodes  and their names the model will start with
                    the names can be any valid python string'''
        bens = []
        mems = []
        lans = []
        motors = []
        world = []

        '''     - the next items describe the edges as a dictionary
                 -> the dictionary entry is always one of the rootnodes, the array following can only contain LANs or LENs

                 !! in case we start from scratch and we rely on peepo to find the best BN -> leave this array empty'''
        edges = []

        '''       - the next items describe the CPD's  as a dictionary
                  -> the dictionary entry is the corresponding node'''
        cpds = []

        '''       - feeding the data'''
        data['Nodes']= {'RONS': {'BENS': bens, 'MEMS': mems}, 'LANS':{'LANS':lans}, 'LENS': {'MOTOR': motors, 'WORLD': world}}
        data['Edges']= edges
        data['CPDs'] = cpds
        return data




def main():
    '''TEMPORARY : for testing purpose  -> will be removed after thourough pratical testing'''

    print("Please enter a valid name for the peepo case i.e. a valid filename without any extension or path.")
    print("If you just want to recreate a slate template, just leave this blank and press ENTER")

    var = input()
    if len(var) == 0:
        var = "Template"
    print("You entered :" + str(var), " OK (Y/N) ?")
    confirm = input()
    util = Utilities(var)
    if confirm == "Y" or confirm ==  "y":
        if  var == "Template":
            util.create_json_template()
            exit()
        else:
            util.create_json_file( description = 'testing')
    ''' going back and forward to test if the get and save methods keep the data integrity'''
    print('expected filename ', util.file)
    networkx, dic, header = util.get_network()
    '''do something with it'''
    util.update_networkx(networkx, dic, header)
    util.save_network()
    network = BayesianModel()
    network, dic, header = util.get_pgmpy_network()
    '''do something with it'''
    util.update_pgmpy(network, dic, header)
    network, dic, header = util.get_pgmpy_network()
    util.save_pgmpy_network()
    ''' backward'''
    networkx, dic, header = util.get_network()
    util.update_networkx(networkx, dic, header)
    util.save_network()
    util.save_pgmpy_network()
    # ''' backward'''
    network = BayesianModel()
    network, dic, header = util.get_pgmpy_network()
    print("Dictionary  ", dic)
    print("header ",  header)

if __name__ == "__main__":
    main()