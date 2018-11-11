from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel

"""
CREATE NEW MODEL
THEN INFER
AFTER WE WILL MODIFY THE MODEL
AND INFER AGAIN

I'm simply creating the same network used here in this tutorial, except that I'm creating the network partially first,
then drawing some inferences on it, and then adding the last node and again querying some inferences:

https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/2.%20Bayesian%20Networks.ipynb
"""

# Initialize model
model = BayesianModel([('D', 'G'), ('I', 'G'), ('G', 'L')])

# Initialize Conditional Probability Tables for all nodes in the network
cpd_d = TabularCPD(variable='D', variable_card=2, values=[[0.6, 0.4]])
cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.7, 0.3]])
cpd_g = TabularCPD(variable='G', variable_card=3,
                   values=[[0.3, 0.05, 0.9, 0.5],
                           [0.4, 0.25, 0.08, 0.3],
                           [0.3, 0.7, 0.02, 0.2]],
                   evidence=['I', 'D'],
                   evidence_card=[2, 2])
cpd_l = TabularCPD(variable='L', variable_card=2,
                   values=[[0.1, 0.4, 0.99],
                           [0.9, 0.6, 0.01]],
                   evidence=['G'],
                   evidence_card=[3])

# Add all the CPDs to the network
# (model is the network object which has references/pointers to all the CPDs, nodes, etc)
model.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l)
# Check model is done to verify that the network is consistent (i.e. CPD values sum to 1 etc.)
model.check_model()

# Perform some inference queries using Variable Elimination (this is an exact inference algorithm, we will need
# to investigate whether an approximate inference algorithm is possible. I haven't found one yet in the library)
infer = VariableElimination(model)
print(infer.query(['G'])['G'])

"""
FROM HERE START CHANGING THE NETWORK 
WE'RE ADDING ONE NODE S WITH AN EDGE COMING FROM I
"""

model.add_node('S')
model.add_edge('I', 'S')

cpd_s = TabularCPD(variable='S', variable_card=2,
                   values=[[0.95, 0.2],
                           [0.05, 0.8]],
                   evidence=['I'],
                   evidence_card=[2])

model.add_cpds(cpd_s)
model.check_model()

infer = VariableElimination(model)
print("========== NEW INFERENCE ============")
print(infer.query(['G'])['G'])
print(infer.query(['S'])['S'])