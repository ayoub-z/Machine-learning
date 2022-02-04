from Perceptron import Perceptron

combinations1 = [[0], [1]]
combinations2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
combinations3 = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
				 [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
class main:

	# AND-gate
	print("\n\n-----AND-GATE-----\n")
	# Initialize the Perceptron
	perceptron1 = Perceptron()

	# Here we are linking neuron 'X1' to neuron 'O'(Output)
	# And again with 'X2' to neuron 'O'
	perceptron1.set_weight('X1', 0.5, 'O')
	perceptron1.set_weight('X2', 0.5, 'O')

	# Set the bias(threshold)
	perceptron1.set_bias(1.0)

	# And finally activate the perceptron for every possible combination
	for i in combinations2:
		perceptron1.activate(i)



	# OR-gate
	print("\n\n-----OR-GATE-----\n")
	perceptron2 = Perceptron()

	perceptron2.set_weight('X1', 0.5, 'O')
	perceptron2.set_weight('X2', 0.5, 'O')

	perceptron2.set_bias(0.5)

	for i in combinations2:
		perceptron2.activate(i)



	# INVERT-gate
	print("\n\n-----INVERT-GATE-----\n")
	perceptron3 = Perceptron()

	perceptron3.set_weight('X1', -1.0, 'O')

	perceptron3.set_bias(-0.5)

	for i in combinations1:
		perceptron3.activate(i)



	# NOR-gate
	print("\n\n-----NOR-GATE-----\n")
	perceptron4 = Perceptron()
	
	perceptron4.set_weight('X1', -1.0, 'O')
	perceptron4.set_weight('X2', -1.0, 'O')
	perceptron4.set_weight('X3', -1.0, 'O')

	perceptron4.set_bias(0)

	for i in combinations3:
		perceptron4.activate(i)			



	# PARTY-PERCEPTRON
	print("\n\n-----PARTY-PERCEPTRON-----\n")
	perceptron5 = Perceptron()

	perceptron5.set_weight('X1', 0.6, 'O')
	perceptron5.set_weight('X2', 0.3, 'O')
	perceptron5.set_weight('X3', 0.2, 'O')

	perceptron5.set_bias(0.4)

	for i in combinations3:
		perceptron5.activate(i)	
