import os
import datetime
import json
import networkx as nx
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
from peepo.visualize.graph import draw_network


class Utilities(object):
    def __init__(self):
        pass

    def translation(dictionary, astring):
        """
        Given an array of tuples (a,b) in dictionary, returns the second element of the tuple where astring was found
        Is used to not loose the users node names as peepo generates standardized names for the corresponding node

        :param dictionary:an array of tuples -> is created in the method : get_network(file)
        :param astring: the name of the node passsed by the user
        :return: the corresponding standardized node name
        :type dictionary: np.array
        :type astring : string
        :rtype : string
        """
        for index, item in enumerate(dictionary):
            if item[0] == astring :
                break
        return  item[1]

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


    def save_pgmpy_network(file, pgmpy_object):
        """
        Saves the passed pgmpy class object in a json file

        """
        a = 1

        """  TO DO """

        return

    def save_network(file,networkx_object):
        """
        Saves the passed networkx class object in a json file

        """
        a = 1

        """  TO DO """

        return

    def get_pgmpy_network(file, pgmpy_object):
        """
        Reads the passed json file and translates it's content to the passed pgmpy class object
        - uses the get_network(file) to read the json file in a networkx format and translate this to pgmpy
        - Creates a dictionary for the nodes in the form of an array of tuples : [(names defines by user, standard name)]

        :param file: : filename without path or extension
        :pgmp_object : the pgmpy object which will be completed
        :return: a dictionary as an array of tuples

        :type file : string
        :type pgmp_object : pgmpy class object
        :rtype : array of tuples
        """
        network, node_dictionary =  Utilities.get_network(file)
        """Iterating through the rootnodes (get's only a 1-dimensional arry"""
        nw_nodes = network.nodes(data = True)

        for i, node in enumerate(nw_nodes):
            parent_nodes = network.predecessors(node[0])
            child_nodes = network.successors(node[0])
            """Add edges"""
            if len(child_nodes) != 0:
                for child in child_nodes:
                    pgmpy_object.add_edge(node[0], child)
            cpd = node[1]['cpd']
            cardinality_parents = []
            for i,nod in enumerate(parent_nodes):
                cpd_nod = network.node[nod]
                cardinality_parents.append(len(cpd_nod['cpd']))
            cardinality_node  = len(cpd)
            if len(cardinality_parents) == 0:
                pgmpy_object.add_cpds(TabularCPD(variable=node[0], variable_card= cardinality_node, values=[cpd]))
                continue
            if len(parent_nodes) == 1:
                cpd_par = network.node[parent_nodes[0]]
                cardinality_parents = [len(cpd_par['cpd'])]
            table = TabularCPD(variable=node[0], variable_card= cardinality_node, values=cpd, \
                              evidence=parent_nodes,\
                              evidence_card=cardinality_parents)
            pgmpy_object.add_cpds(table)

        # pgmpy_object.check_model()
        # draw_network(pgmpy_object)

        return node_dictionary


    def get_network(file):
        """
        Reads the passed json file and translate it's content in a networkx class object
        - The nodes in the object are renamed so they have a standardized signature
        - Creates a dictionary for the nodes in the form of an array of tuples : [(names defines by user, standard name)]

        :param file: : filename without path or extension
        :return: a networkx class object and dictionary as an array of tuples

        :type file : string
        :rtype : networkx class object, array of tuples
        """
        with open(Utilities.get_json_path(file)) as f:
            data = json.load(f)
        G = nx.DiGraph()

        '''Feeding G with the nodes'''
        nodes = []
        node_dictionary = []
        for key in data['Nodes'].keys():
            for  secondkey in data['Nodes'][key].keys():
                for c, n  in enumerate(data['Nodes'][key][secondkey]):
                    node = secondkey + "_" + str(c)
                    nodes.append(node)
                    node_dictionary.append((n,node))
        np.ravel(nodes)
        G.add_nodes_from(nodes)

        '''Feeding G with the edges'''
        edges = []
        for j, pair in enumerate(data['Edges']):
            for parent in pair.keys():
                for child in data['Edges'][j][parent]:
                    parent_ = Utilities.translation(node_dictionary,parent)
                    child_  = Utilities.translation(node_dictionary,child)
                    edges.append((parent_,child_))
        np.ravel(edges)
        G.add_edges_from(edges)

        '''Feeding G with the CPD's as nodes attributes'''
        for j, node in enumerate(data['CPDs']):
            for parent in node.keys():
                node_ = Utilities.translation(node_dictionary, parent)
                cpds = {'cpd':data['CPDs'][j][parent]}
                G.add_node(node_ ,cpds)

        return G, node_dictionary


    def create_json_file(case_name, identification_ = None, description_ = None, frozen_ = None, nodes_ = None, edges_ = None, cpds_ = None, train_from_ = None):
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
        Case_name = Utilities.get_json_path(case_name)  # creates the right path in which case_name will be saved
        data = Utilities.get_empty_canvas()

        '''       - the 3 next items are for tracking purpose only, not fundamentally necessary'''
        data['Identificaton'] = case_name
        data['Date'] = datetime.datetime.now().strftime("%c")
        data['Description'] = 'Blablabla'
        '''       - the next item gives a file containing possible training data (OPTIONAL)'''
        data['Train_from'] = 'None'

        '''      Frozen tells whether or not the model can be considered as final i.e. is there still "training" needed'''
        data['Frozen'] = False

        '''       - the 5 next lines tells how much nodes  and their names the model will start with
                    the names can be any valid python string'''
        bens = ['pooping', 'peeing', 'constipated']
        mems = ['havenotoiletpaper']
        lans = ['diarhea', 'happypoop']
        motors = ['asshole1', 'asshole2']
        world = ['toilet1', 'toilet2', 'garden1', 'garden2','doctor']

        '''     - the next items describe the edges as a dictionary
                 -> the dictionary entry is always one of the rootnodes, the array following can only contain LANs or LENs'''
        edges = []

        '''       !! in case we start from scratch and we rely on peepo to find the best BN -> leave this array empty'''
        edges.append({'pooping': ['toilet1', 'diarhea','happypoop']})
        edges.append({'peeing': ['toilet2', 'garden1', 'garden2']})
        edges.append({'constipated': ['doctor']})
        edges.append({'havenotoiletpaper': ['garden1', 'garden2']})
        edges.append({'diarhea': ['toilet1', 'doctor']})
        edges.append({'happypoop': ['garden1', 'garden2']})
        edges.append({'diarhea': ['asshole1', 'asshole2']})
        edges.append({'happypoop': ['asshole1', 'asshole2']})


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
        data["Nodes"]['LANS']['LANS']  = lans
        data["Nodes"]['LENS']['MOTOR'] = motors
        data["Nodes"]['LENS']['WORLD'] = world
        data["Edges"] = edges
        data["CPDs"] = cpds


        ''' dumping to Case_name file in jason format'''
        with open(Case_name, 'w') as f:
            json.dump(data, f, indent=3)

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
        ''' dumping to CASENAME file in jason format'''
        with open(case_name, 'w') as f:
            json.dump(data, f, indent=3)
        print("Empty template created")

    def get_empty_canvas():
        """
         This method creates a json canvas which will be used for the several json creating method

         :param : void
         :return: a dictionary with the structure of the json file
         :type : non
         :rtype : dictionary
         """

        data = {'Identificaton': '', 'Date': '', 'Description': '', 'Frozen': '', 'Nodes': [], 'Edges': [], 'CPDs': [],
                'Train_from': ''}

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
        data['Nodes']= {'RONS': {'BENS': bens, 'MEMS': mems}, 'LANS':{'LANS': lans}, 'LENS': {'MOTOR': motors, 'WORLD': world}}
        data['Edges'].append(edges)
        data['CPDs'].append(cpds)
        return data

def main():
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
    network = BayesianModel()
    dic = Utilities.get_pgmpy_network(var, network)
if __name__ == "__main__":
    main()