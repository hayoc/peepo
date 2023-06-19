import functools

import jax
import matplotlib.pyplot as plt
import numpy as np
import itertools
import math

############
# Load PGMax
from pgmax import fgraph, fgroup, infer, vgroup, factor


root_names = ("a")
leaf_names = ("b")


# hidden_variables = vgroup.NDVarArray(num_states=2, shape=(1,))
# visible_variables = vgroup.NDVarArray(num_states=2, shape=(1,))

root_variables = vgroup.VarDict(num_states=2, variable_names=root_names)
leaf_variables = vgroup.VarDict(num_states=2, variable_names=leaf_names)
fg = fgraph.FactorGraph(variable_groups=[root_variables, leaf_variables])

# r1_factor = factor.EnumFactor(
#       variables=[root_variables["a"]],
#       factor_configs=np.array([[0], [1]]),
#       log_potentials=np.array([math.log(0.9), math.log(0.1)]),
# )
# fg.add_factors(r1_factor)

factor_group = fgroup.EnumFactorGroup(
    variables_for_factors=[[root_variables['a']]],
    factor_configs=np.arange(2)[:, None],
    log_potentials=np.array([math.log(0.9), math.log(0.1)])
)
fg.add_factors(factor_group)

pairwise_factor = factor.EnumFactor(
    variables=[root_variables["a"], leaf_variables["b"]],
    factor_configs=np.array(list(itertools.product(np.arange(2), repeat=2))),
    log_potentials=np.array([math.log(0.3), math.log(0.7), math.log(0.9), math.log(0.1)]),
)
fg.add_factors(pairwise_factor)

bp = infer.build_inferer(fg.bp_state, backend="bp")
bp_arrays = bp.init()


bp_arrays = bp.run(bp_arrays, num_iters=100, damping=0.5, temperature=0.0)
beliefs = bp.get_beliefs(bp_arrays)
marginals = infer.get_marginals(beliefs)

print(marginals[root_variables])
print(marginals[leaf_variables])

#bp_arrays = bp.update(log_potentials_updates={factor_group: np.array([math.log(0.1), math.log(0.9)])})
bp_arrays = bp.update(evidence_updates={root_variables['a']: np.array([0., 4.5])})

bp_arrays = bp.run(bp_arrays, num_iters=100, damping=0.5, temperature=0.0)
beliefs = bp.get_beliefs(bp_arrays)
marginals = infer.get_marginals(beliefs)

print(marginals[root_variables])
print(marginals[leaf_variables])



