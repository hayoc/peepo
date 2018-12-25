import math
from pomegranate import *
import numpy as np
#Let's create the distributions for the guest and the prize. Note that both distributions are independent of one another.

guest = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )
prize = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )
#Now let's create the conditional probability table for our Monty. The table is dependent on both the guest and the prize.

monty = ConditionalProbabilityTable(
	[[ 'A', 'A', 'A', 0.0 ],
	 [ 'A', 'A', 'B', 0.5 ],
	 [ 'A', 'A', 'C', 0.5 ],
	 [ 'A', 'B', 'A', 0.0 ],
	 [ 'A', 'B', 'B', 0.0 ],
	 [ 'A', 'B', 'C', 1.0 ],
	 [ 'A', 'C', 'A', 0.0 ],
	 [ 'A', 'C', 'B', 1.0 ],
	 [ 'A', 'C', 'C', 0.0 ],
	 [ 'B', 'A', 'A', 0.0 ],
	 [ 'B', 'A', 'B', 0.0 ],
	 [ 'B', 'A', 'C', 1.0 ],
	 [ 'B', 'B', 'A', 0.5 ],
	 [ 'B', 'B', 'B', 0.0 ],
	 [ 'B', 'B', 'C', 0.5 ],
	 [ 'B', 'C', 'A', 1.0 ],
	 [ 'B', 'C', 'B', 0.0 ],
	 [ 'B', 'C', 'C', 0.0 ],
	 [ 'C', 'A', 'A', 0.0 ],
	 [ 'C', 'A', 'B', 1.0 ],
	 [ 'C', 'A', 'C', 0.0 ],
	 [ 'C', 'B', 'A', 1.0 ],
	 [ 'C', 'B', 'B', 0.0 ],
	 [ 'C', 'B', 'C', 0.0 ],
	 [ 'C', 'C', 'A', 0.5 ],
	 [ 'C', 'C', 'B', 0.5 ],
	 [ 'C', 'C', 'C', 0.0 ]], [guest, prize] )

#Now lets create the states for the bayesian network.

s1 = State( guest, name="guest" )
s2 = State( prize, name="prize" )
s3 = State( monty, name="monty" )
#Then the bayesian network itself, adding the states in after.

network = BayesianNetwork( "test" )
network.add_states( s1, s2, s3 )
#Then the transitions.

network.add_transition( s1, s3 )
network.add_transition( s2, s3 )
#
# print(s1)
# print(s2)
# print(s3)
network.bake()
#Now we can train our network on the following data.

data = [[ 'A', 'A', 'A' ],
		[ 'A', 'A', 'A' ],
		[ 'A', 'A', 'A' ],
		[ 'A', 'A', 'A' ],
		[ 'A', 'A', 'A' ],
		[ 'B', 'B', 'B' ],
		[ 'B', 'B', 'C' ],
		[ 'C', 'C', 'A' ],
		[ 'C', 'C', 'C' ],
		[ 'C', 'A', 'C' ],
		[ 'C', 'C', 'C' ],
		[ 'C', 'A', 'A' ]]

data_ = [[ 'A', 'A', 'A' ]
]

labels =np.asarray([ 'A', 'A', 'A' ])
network.fit( data )
#Now let's see what happens when our Guest says 'A' and the Prize is 'A'.
score = network.score(data_, labels)
print(score)
# observations = { 'guest' : 'A', 'prize' : 'A' }
# beliefs = map( str, network.predict_proba( observations ) )
# print( "\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) ))
#
#
# model = NaiveBayes( MultivariateGaussianDistribution, n_components=2 )
# X = np.array([[ 6, 180, 12 ],
#               [ 5.92, 190, 11 ],
#               [ 5.58, 170, 12 ],
#               [ 5.92, 165, 10 ],
#               [ 6, 160, 9 ],
#               [ 5, 100, 6 ],
#               [ 5.5, 100, 8 ],
#               [ 5.42, 130, 7 ],
#               [ 5.75, 150, 9 ],
#               [ 5.5, 140, 8 ]])
#
# y = np.array([ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 ])
#
# model.fit( X, y )
# print(model.score(X,y))
# # data = np.array([[ 5.75, 130, 8 ]])
#
# for sample, probs in zip( data, model.predict_proba( data ) ):
#     print "Height {}, weight {}, and foot size {} is {:.3}% male, {:.3}% female.".format( sample[0], sample[1], sample[2], 100*probs[0], 100*probs[1] )
#
#
# for sample, result in zip( data, model.predict( data ) ):
#     print "Person with height {}, weight {}, and foot size {} is {}".format( sample[0], sample[1], sample[2], "female" if result else "male" )