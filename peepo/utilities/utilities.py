import os
import datetime
import json
import networkx as nx
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
import re
from peepo.visualize.graph import draw_network
import matplotlib.pyplot as plt

import difflib

class Utilities(object):
    def __init__(self):
        pass

    def  get_keywords():
        return  ['BENS','MEMS','LANS','MOTOR','WORLD']

    def save_json(file, astring):
        """
        This helping function is only needed to have the json file be formatted in a user friendly way
        as the "dump" method does not provide a lot of possibilities to get it "pretty"

        :param file :the ull path of the json file
        :param astring: the name of the string containing the whole information
        :return: void
        :type file: string
        :type astring : string
        :rtype : void
        """
        text_file = open(file, "w")
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

    def translation(dictionary, astring, from_man_to_machine):
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

        for index, item in enumerate(dictionary):
            if item[source] == astring :
                break
        return  item[target]

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

    def get_json_path(filename):
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
            if  'peepo\peepo' not in common:
                break
        Utilities.check_json_path(common)
        return common + '\project_repository\\'+filename+'.json'


    def save_pgmpy_network(file, header, dictionary, pgmpy_object):
        """
                Saves the passed pgmpy_object class object in a json file

                """
        data = Utilities.get_empty_canvas()
        data["header"] = header
        keywords = Utilities.get_keywords()

        nw_nodes = pgmpy_object.nodes(data=True)
        nodes = {'RONS':{'BENS':[],'MEMS':[]},'LANS':{'LANS':[]},'LENS':{'MOTOR':[],'WORLD':[]}}
        edges = []
        cpds = []
        for i, node in enumerate(nw_nodes):
            node_name = node[0]
            edge_ = np.ravel(pgmpy_object.edges(node_name))
            edge = []
            for k, n in enumerate(edge_):
                if n != node_name:
                    edge.append(n)
            print("edges from node ", node_name, " = ", edge)
            cardinality = int(pgmpy_object.get_cardinality(node_name))
            cpd = pgmpy_object.get_cpds(node_name).values
            cpd = cpd.astype(int)
            print('done')
            placeholder = ''
            edge_ = []
            for e, ed in enumerate(edge):
                for pseudonym in keywords:
                    if pseudonym in ed:
                        edge_.append(Utilities.translation(dictionary, ed, 1))
            for pseudonym in keywords:
                if pseudonym in node_name:
                    placeholder = pseudonym
                    node_name = Utilities.translation(dictionary, node_name, 1)
                    if placeholder == 'BENS' or placeholder == 'MEMS':
                        nodes['RONS'][placeholder].append([node_name, cardinality])
                    if placeholder == 'LANS':
                        nodes['LANS'][placeholder].append([node_name, cardinality])
                    if placeholder == 'MOTOR' or placeholder == 'WORLD':
                        nodes['LENS'][placeholder].append([node_name, cardinality])
            if len(edge_) != 0:
                edges.append({node_name: edge_})
            mat = np.asarray(cpd .ndim)
            cpd_ = []
            if mat == 1:
                cpd_ = cpd.tolist()
            if mat != 1:
                mat = np.shape(cpd)
                for row in range(0, mat[0]):
                    new_row = cpd[row,:].ravel()
                    cpd_.append(new_row.tolist())
            cpds.append({node_name: cpd_})
        data['Nodes'] = nodes
        data['Edges'] = edges
        data['CPDs'] = cpds
        data['header']['Date'] = datetime.datetime.now().strftime("%c")
        Utilities.save_json(Utilities.get_json_path(file), json.dumps(data))
        return

    def save_network(file,header, dictionary,networkx_object):
        """
        Saves the passed networkx class object in a json file

        """
        data = Utilities.get_empty_canvas()
        data["header"] = header
        keywords = Utilities.get_keywords()
        nw_nodes = networkx_object.nodes(data = True)
        nodes = {'RONS': {'BENS': [], 'MEMS': []}, 'LANS': {'LANS': []}, 'LENS': {'MOTOR': [], 'WORLD': []}}
        edges = []
        cpds = []
        for i, node in enumerate(nw_nodes):
            node_name = node[0]
            edge = np.ravel(networkx_object.edges(node_name))
            edge = edge[edge != node_name]
            cardinality = node[1]['cardinality']
            cpd = node[1]['cpd']
            placeholder =''
            edge_ = []
            for e, ed in enumerate(edge):
                for pseudonym  in keywords:
                    if pseudonym in ed:
                        edge_.append(Utilities.translation(dictionary, ed,1))
            for pseudonym  in keywords:
                if pseudonym in node_name:
                    placeholder = pseudonym
                    node_name = Utilities.translation(dictionary, node_name,1)
                    if placeholder == 'BENS' or placeholder == 'MEMS':
                        nodes['RONS'][placeholder].append([node_name, cardinality])
                    if placeholder == 'LANS':
                        nodes['LANS'][placeholder].append([node_name, cardinality])
                    if placeholder == 'MOTOR' or placeholder == 'WORLD':
                        nodes['LENS'][placeholder].append([node_name, cardinality])
            if len(edge_) != 0:
                edges.append({node_name: edge_})
            cpds.append({node_name:cpd})
        data['Nodes'] = nodes
        data['Edges'] = edges
        data['CPDs']  = cpds
        data['header']['Date'] = datetime.datetime.now().strftime("%c")
        Utilities.save_json(Utilities.get_json_path(file) , json.dumps(data))
        return

    def get_pgmpy_network(file, pgmpy_object):
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
        """
        network, node_dictionary, header =  Utilities.get_network(file)
        """Iterating through the rootnodes (get's only a 1-dimensional arry"""
        nw_nodes = network.nodes(data = True)

        for i, node in enumerate(nw_nodes):
            pgmpy_object.add_node(node[0])

        for i, node in enumerate(nw_nodes):
            parent_nodes = network.predecessors(node[0])
            child_nodes = network.successors(node[0])
            print('childenodes for node ', node[0], ' =' , child_nodes)
            """Add nodes and edges in one loop"""
            if len(child_nodes) != 0:
                for child in child_nodes:
                    pgmpy_object.add_edge(node[0], child)
            # if len(child_nodes) != 0:
            #     pgmpy_object.add_edge(node[0], np.asarray(child_nodes))
            """ add cpd's"""
            cpd = node[1]['cpd']
            ''' find the cardinality of the node and it's parents'''
            cardinality_node = node[1]['cardinality']
            cardinality_parents = []
            for i,nod in enumerate(parent_nodes):
                attribute = network.node[nod]
                cardinality_parents.append(attribute['cardinality'])
            ''' Depending on the place in the BN and/or the number of parents  the CPD have another call'''
            if len(cardinality_parents) == 0:
                pgmpy_object.add_cpds(TabularCPD(variable=node[0], variable_card= cardinality_node, values=[cpd]))
                continue
            if len(parent_nodes) == 1:#this is necessary so the TabularCPD function accepts only one parent (evidence_card must be reshaped)
                attribute = network.node[parent_nodes[0]]
                cardinality_parents = [attribute['cardinality']]
            table = TabularCPD(variable=node[0], variable_card= cardinality_node, values=cpd, \
                              evidence=parent_nodes,\
                              evidence_card=cardinality_parents)
            pgmpy_object.add_cpds(table)

        '''------TO DELETE-------------'''
        # pgmpy_object.check_model()
        # draw_network(pgmpy_object)
        '''-----------------------------'''
        return node_dictionary, header


    def get_network(file):
        """
        Reads the passed json file and translate it's content in a networkx class object
        - The nodes in the object are renamed so they have a standardized signature
        - Creates a dictionary for the nodes in the form of an array of tuples : [(names defines by user, standard name)]

        :param file: : filename without path or extension
        :return: a networkx class object, dictionary as an array of tuples and the header of the json file

        :type file : string
        :rtype : networkx class object, array of tuples, dictionary
        """
        with open(Utilities.get_json_path(file)) as f:
            data = f.read()
        data = re.sub('\n', '', data)
        data = re.sub('\t', '', data)
        data = json.loads(data)

        header = data['header']
        G = nx.DiGraph()

        '''Feeding G with the nodes'''
        nodes = []
        node_dictionary = []
        cardinality = {}
        for key in data['Nodes'].keys():
            for  secondkey in data['Nodes'][key].keys():
                for c, n  in enumerate(data['Nodes'][key][secondkey]):
                    node = secondkey + "_" + str(c)
                    nodes.append(node)
                    node_dictionary.append((n[0],node))
                    cardinality.update({node:n[1]})#this contains the cardinality of each node with the node name as dictionary entry
        np.ravel(nodes)
        G.add_nodes_from(nodes)

        '''Feeding G with the edges'''
        edges = []
        for j, pair in enumerate(data['Edges']):
            for parent in pair.keys():
                for child in data['Edges'][j][parent]:
                    parent_ = Utilities.translation(node_dictionary,parent, 0)
                    child_  = Utilities.translation(node_dictionary,child, 0)
                    edges.append((parent_,child_))
        np.ravel(edges)
        G.add_edges_from(edges)

        '''Feeding G with the cardinality and CPD's as nodes attributes'''
        for j, node in enumerate(data['CPDs']):
            for parent, cpd in node.items():
                node_ = Utilities.translation(node_dictionary, parent, 0)
                addendum = {'cardinality': cardinality[node_], 'cpd': cpd}
                G.add_node(node_ ,addendum)

        '''TO REMOVE LATER'''
        # plt.figure(figsize=(10, 5))
        # pos = nx.circular_layout(G, scale=2)
        # node_labels = nx.get_node_attributes(G, 'cpd')
        # nx.draw(G, pos, node_size=1200, node_color='lightblue',
        #         linewidths=0.25,  font_size=10, font_weight='bold', with_labels=True)
        # plt.show()
        return G, node_dictionary, header


    def create_json_file(Case_name, identification_ = None, description_ = None, frozen_ = None, nodes_ = None, edges_ = None, cpds_ = None, train_from_ = None):

        """
        A helping method if the user prefers to create the BN within the code

        :param case_name: the file name without path or extension where the json file will be saved
        :param :
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
        case_name = Utilities.get_json_path(Case_name)  # creates the right path in which case_name will be saved
        #json_tab_file_write = JSONTabIndentFileWriter( Case_name,5a)
        data = Utilities.get_empty_canvas()

        '''       - the 3 next items are for tracking purpose only, not fundamentally necessary'''
        data["header"]['Identification'] = case_name
        data["header"]['Date'] = datetime.datetime.now().strftime("%c")
        data["header"]['Description'] = 'Blablabla'
        '''       - the next item gives a file containing possible training data (OPTIONAL)'''
        data["header"]['Train_from'] = 'None'

        '''      Frozen tells whether or not the model can be considered as final i.e. is there still "training" needed'''
        data["header"]['Frozen'] = False

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

        # json.dump(data,json_tab_file_write, indent=5, separators=('}', ': '), sort_keys=False)
        # # close file writer
        # json_tab_file_write.close()
        ''' dumping to CASENAME file in jason format'''
        # with open(case_name, 'w') as f:
        #     json.dump(data, f, separators = (",",":"))
        Utilities.save_json(case_name,json.dumps(data))

        print("Json file for  - ", case_name, "  - created")

    def create_json_template():
        """
        A helping method if the  jason template in the project_repository ditectory has been deleted or corrupted

        :param : void
        :return: void

        :type : void
        :rtype : void
        """
        case_name = "Template"
        case_name = Utilities.get_json_path(case_name)  # creates the right path in which case_name will be saved
        data = Utilities.get_empty_canvas()
        data['header']['Identification'] = case_name
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
        Utilities.save_json(case_name,json.dumps(data))
        print("Empty template created")

    def get_empty_canvas():
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
    '''TEMPORARY : for testing purpose'''
    print("Please enter a valid name for the peepo case i.e. a valid filename without any extension or path.")
    print("If you just want to recreate a slate template, just leave this blank and press ENTER")
    var = input()
    if len(var) == 0:
        var = "Template"
    print("You entered :" + str(var), " OK (Y/N) ?")
    confirm = input()

    if confirm == "Y" or confirm ==  "y":
        if  var == "Template":
            Utilities.create_json_template()
        else:
            Utilities.create_json_file(str(var))
    networkx, dic, header = Utilities.get_network(var)
    # print("Dictionary  ", dic)
    # print("header ",  header)
    network = BayesianModel()
    dic, header = Utilities.get_pgmpy_network(var, network)
    Utilities.save_network(var, header,dic,networkx)
    # ''' backward'''
    networkx, dic, header = Utilities.get_network(var)


    Utilities.save_pgmpy_network(var, header, dic, network)
    ''' backward'''
    network = BayesianModel()
    dic, header = Utilities.get_pgmpy_network(var, network)


if __name__ == "__main__":
    main()