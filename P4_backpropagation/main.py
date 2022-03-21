from Neuron import Neuron
from NeuronLayer import NeuronLayer
from NeuronNetwork import NeuronNetwork

AND_data = {'data':[[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]], 'target':[0, 0, 0, 1]}

XOR_data = {'data':[[0, 0],
					[0, 1],
					[1, 0],
					[1, 1]], 'target':[0, 1, 1, 0]}				

HALF_ADDER = {'data':[[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]], 'target':[[0, 0], [1, 0], [1, 0], [0, 1]]}

if __name__ == "__main__":
	# AND-gate
	print("\n\n-----AND-GATE-----\n")	
	neuron_network1 = NeuronNetwork()
	neuron_network1.create_layers(2,[], 1)

	neuron_network1.train(AND_data['data'], AND_data['target'], l_rate=0.5, epochs=10000)
	neuron_network1.__str__(AND_data['data'], testing=True)


	# Half adder
	print("\n\n-----Half adder-----\n")
	neuron_network2 = NeuronNetwork()
	neuron_network2.create_layers(2,[3], 2)

	neuron_network2.train(HALF_ADDER['data'], HALF_ADDER['target'], l_rate=0.5, epochs=10000)
	neuron_network2.__str__(HALF_ADDER['data'], testing=True)

	# XOR-gate 
	print("\n\n-----XOR-GATE-----\n")	
	neuron_network3 = NeuronNetwork()
	neuron_network3.create_layers(2,[2], 1)

	neuron_network3.train(XOR_data['data'], XOR_data['target'], l_rate=0.5, epochs=10000)
	neuron_network3.__str__(XOR_data['data'], testing=True)