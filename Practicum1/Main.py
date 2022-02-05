from Perceptron import Perceptron

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
