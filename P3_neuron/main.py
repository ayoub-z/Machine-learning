from Neuron import Neuron
from NeuronLayer import NeuronLayer
from NeuronNetwork import NeuronNetwork

combinations1 = [[0], [1]]
combinations2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
combinations3 = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
				 [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

if __name__ == "__main__":

	# We will NOT adjust the waits for networks that only have a single layer,
	# Since there is only one output and that output already gets rounded to a 0 or 1
	# The weights/bias of the ADDER/XOR however we do adjust, as you'll see later on

	# AND-gate
	print("\n\n-----AND-GATE-----\n")
	# Initialize the Neuron
	Neuron1 = Neuron()

	# Set the weights
	Neuron1.set_weight([0.5, 0.5])
	# Set the bias(threshold)
	Neuron1.set_bias(-1.0)

	# And finally activate the Neuron for every possible combination
	# and print the result
	for i in combinations2:
		Neuron1.activate(i)
		Neuron1.__str__(i)



	# OR-gate
	print("\n\n-----OR-GATE-----\n")
	Neuron2 = Neuron()

	Neuron2.set_weight([0.5, 0.5])
	Neuron2.set_bias(-0.5)

	for i in combinations2:
		Neuron2.activate(i)
		Neuron2.__str__(i)



	# INVERT-gate
	print("\n\n-----INVERT-GATE-----\n")
	Neuron3 = Neuron()

	Neuron3.set_weight([-1.0])
	Neuron3.set_bias(0.5)

	for i in combinations1:
		Neuron3.activate(i)
		Neuron3.__str__(i)



	# NOR-gate
	print("\n\n-----NOR-GATE-----\n")
	Neuron4 = Neuron()

	Neuron4.set_weight([-1.0, -1.0, -1.0])
	Neuron4.set_bias(0)

	for i in combinations3:
		Neuron4.activate(i)			
		Neuron4.__str__(i)



	# PARTY-Neuron. Figure 2.8 from reader
	print("\n\n-----PARTY-Neuron-----\n")
	Neuron5 = Neuron()

	Neuron5.set_weight([0.6, 0.3, 0.2])
	Neuron5.set_bias(-0.4)

	for i in combinations3:
		Neuron5.activate(i)	
		Neuron5.__str__(i)



	# Testing NeuronLayer with AND-gate 
	print("\n\n-----AND-GATE NeuronLayer-----\n")
	# Initialize the NeuronLayer
	Neuron_layer1 = NeuronLayer()

	# Initialize the Neurons
	Neuron_layer1.create_Neurons(2, 1)
	# Set the weights
	Neuron_layer1.set_weight([1, 1]) 
	# Set the bias(threshold)
	Neuron_layer1.set_bias([-2])

	for i in combinations2:
		Neuron_layer1.activate(i)
		Neuron_layer1.__str__(i)



	# XOR-gate 
	print("\n\n-----XOR-GATE-----\n")	
	Neuron_network1 = NeuronNetwork()

	Neuron_network1.create_layers(2,[2],1)
	Neuron_network1.set_weight([[[100, -100], [-100, 100]], [100, 100]])
	Neuron_network1.set_bias([[-100, -100], [-50]])

	for i in combinations2:
		Neuron_network1.activate(i)
		Neuron_network1.__str__(i)



	# Half adder
	print("\n\n-----Half adder-----\n")
	Neuron_network2 = NeuronNetwork()

	Neuron_network2.create_layers(2,[2], 2)
	Neuron_network2.set_weight([[[100, 100], [100, 100]], [[100, 0], [-100, 100]]])
	Neuron_network2.set_bias([[-50, -100], [-50, -100]])

	for i in combinations2:
		Neuron_network2.activate(i)
		Neuron_network2.__str__(i)	