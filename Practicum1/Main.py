from Perceptron import Perceptron
from PerceptronLayer import PerceptronLayer
from PerceptronNetwork import PerceptronNetwork

combinations1 = [[0], [1]]
combinations2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
combinations3 = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
				 [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

if __name__ == "__main__":

	# AND-gate
	print("\n\n-----AND-GATE-----\n")
	# Initialize the Perceptron
	perceptron1 = Perceptron()

	# Set the weights
	perceptron1.set_weight([0.5, 0.5])
	# Set the bias(threshold)
	perceptron1.set_bias(1.0)

	# And finally activate the perceptron for every possible combination
	# and print the result
	for i in combinations2:
		perceptron1.activate(i)
		perceptron1.__str__(i)



	# OR-gate
	print("\n\n-----OR-GATE-----\n")
	perceptron2 = Perceptron()

	perceptron2.set_weight([0.5, 0.5])
	perceptron2.set_bias(0.5)

	for i in combinations2:
		perceptron2.activate(i)
		perceptron2.__str__(i)



	# INVERT-gate
	print("\n\n-----INVERT-GATE-----\n")
	perceptron3 = Perceptron()

	perceptron3.set_weight([-1.0])
	perceptron3.set_bias(-0.5)

	for i in combinations1:
		perceptron3.activate(i)
		perceptron3.__str__(i)



	# NOR-gate
	print("\n\n-----NOR-GATE-----\n")
	perceptron4 = Perceptron()

	perceptron4.set_weight([-1.0, -1.0, -1.0])
	perceptron4.set_bias(0)

	for i in combinations3:
		perceptron4.activate(i)			
		perceptron4.__str__(i)



	# PARTY-PERCEPTRON
	print("\n\n-----PARTY-PERCEPTRON-----\n")
	perceptron5 = Perceptron()

	perceptron5.set_weight([0.6, 0.3, 0.2])
	perceptron5.set_bias(0.4)

	for i in combinations3:
		perceptron5.activate(i)	
		perceptron5.__str__(i)



	# Testing PerceptronLayer with AND-gate 
	print("\n\n-----AND-GATE PerceptronLayer-----\n")
	# Initialize the PerceptronLayer
	perceptron_layer1 = PerceptronLayer()

	# Initialize the Perceptrons
	perceptron_layer1.create_perceptrons(2, 1)
	# Set the weights
	perceptron_layer1.set_weight([1, 1]) 
	# Set the bias(threshold)
	perceptron_layer1.set_bias([2])

	for i in combinations2:
		perceptron_layer1.activate(i)
		perceptron_layer1.__str__(i)



	# XOR-gate 
	print("\n\n-----XOR-GATE-----\n")	
	perceptron_network1 = PerceptronNetwork()

	perceptron_network1.create_layers(2,[2],1)
	perceptron_network1.set_weight([[[1, -1], [-1, 1]], [1, 1]])
	perceptron_network1.set_bias([[1, 1], [1]])

	for i in combinations2:
		perceptron_network1.activate(i)
		perceptron_network1.__str__(i)



	# Half adder
	print("\n\n-----Half adder-----\n")
	perceptron_network2 = PerceptronNetwork()

	perceptron_network2.create_layers(2,[2], 2)
	perceptron_network2.set_weight([[[1, 1], [1, 1]], [[1, 0], [-1, 1]]])
	perceptron_network2.set_bias([[1, 2], [1, 1]])

	for i in combinations2:
		perceptron_network2.activate(i)
		perceptron_network2.__str__(i)