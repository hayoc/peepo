import os
import datetime
import json


class Utilities(object):
    def __init__(self):
        pass

    def check_dir(directory):
        if not os.path.exists(directory + '\project_repository\\'):
            os.makedirs(directory + '\project_repository\\')

    def get_json_path(filename):
        levels = 5
        filepath = os.path.dirname(os.path.realpath(__file__))
        common = filepath
        for i in range(levels + 1):
            common = os.path.dirname(common)
            if  'peepo\peepo' not in common:
                break
        Utilities.check_dir(common)
        return common + '\project_repository\\'+filename+'.json'

    def create_json_file(case_name, identification_ = None, description_ = None, frozen_ = None, nodes_ = None, edges_ = None, cpds_ = None, train_from_ = None):

        Case_name = Utilities.get_json_path(case_name)  # creates the right path in which case_name will be saved

        '''     data is a dictionary which will be saved in json format
                    - initialize the dictionary
                    '''

        data = {'Identificaton': '', 'Date': '', 'Description': '', 'Frozen': '', 'Nodes': [], 'Edges': [], 'CPDs': [],
                'Train_from': ''}

        '''       - the 3 next items are for tracking purpose only, not fundamentally necessary'''

        data['Identificaton'] = case_name
        data['Date'] = datetime.datetime.now().strftime("%c")
        data['Description'] = 'Blablabla'
        '''       - the next item gives a file containing possible training data (OPTIONAL)'''
        data['Train_from'] = 'a_filename_without_extension'

        '''      Frozen tells whether or not the model can be considered as final i.e. is there still "training" needed'''

        data['Frozen'] = False

        '''       - the 5 next lines tells how much nodes  and their names the model will start with
                    the names can be any valid python string'''

        bens = ['ben1', 'ben2', 'ben3']
        mems = ['mem1']
        lans = []
        motors = ['motor1', 'motor2']
        world = ['wor1d1', 'world2', 'world3', 'world4']

        '''     - the next items describe the edges as a dictionary
                 -> the dictionary entry is always one of the parents, the array following can only contain LANs or LENs'''

        edges = []

        '''       !! in case we start from scratch and we rely on peepo to find the best BN -> leave this array empty'''

        edges.append({'ben1': ['motor1', 'world2', 'world3']})
        edges.append({'ben2': ['motor1', 'world1', 'world3']})
        edges.append({'ben3': ['motor2', 'world1', 'world2']})
        edges.append({'mem1': ['world1', 'world2', 'world3', 'world4']})

        '''       - the next items describe the CPD's  as a dictionary
                  -> the dictionary entry is the corresponding node'''

        cpds = []  # ?? how to format the CPD ??

        '''       - feeding the data'''
        data['Nodes'].append(
            {'PARS': {'BENS': bens, 'MEMS': mems}, 'LANS': lans, 'LENS': {'MOTOR': motors, 'WORLD': world}})
        data['Edges'].append(edges)
        data['CPDs'].append(cpds)

        ''' dumping to Case_name file in jason format'''
        with open(Case_name, 'w') as f:
            json.dump(data, f, indent=3)

        print("Json file for  - ", case_name, "  - created")

    def create_json_template():
        case_name = "Template"
        case_name = Utilities.get_json_path(case_name)  # creates the right path in which case_name will be saved
        print("Full Case name = ", case_name)
        '''     data is a dictionary which will be saved in json format
                    - initialize the dictoionary
                    '''

        data = {'Identificaton': '', 'Date': '', 'Description': '', 'Frozen': '', 'Nodes': [], 'Edges': [], 'CPDs': [],
                'Train_from': ''}

        '''       - the 3 next items are for tracking purpose only, not fundamentally necessary'''

        data['Identificaton'] = ''
        data['Date'] = ''
        data['Description'] = ''
        '''       - the next items gives a file containing possible training data (OPTIONAL)'''
        data['Train_from'] = ''

        '''      Frozen tells whether or not the model can be considered as final i.e. is there still "training" needed'''

        data['Frozen'] = False

        '''       - the 5 next lines tells how much nodes  and their names the model will start with
                    the names can be any valid python string'''

        bens = []
        mems = []
        lans = []
        motors = []
        world = []

        '''     - the next items describe the edges as a dictionary
                 -> the dictionary entry is always one of parents, the array following can only contain LANs or LENs'''

        edges = []

        '''       !! in case we start from scratch and we rely on peepo to find the best BN -> leave this array empty'''

        edges.append({'dummy': ['dummy1', 'dummyn']})

        '''       - the next items describe the CPD's  as a dictionary
                  -> the dictionary entry is the corresponding node'''

        cpds = []  # ?? how to format the CPD ??

        '''       - feeding the data'''
        data['Nodes'].append(
            {'PARS': {'BENS': bens, 'MEMS': mems}, 'LANS': lans, 'LENS': {'MOTOR': motors, 'WORLD': world}})
        data['Edges'].append(edges)
        data['CPDs'].append(cpds)

        ''' dumping to CASENAME file in jason format'''
        with open(case_name, 'w') as f:
            json.dump(data, f, indent=3)

        print("Empty template created")

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

if __name__ == "__main__":
    main()