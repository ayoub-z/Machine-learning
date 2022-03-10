from NeuronLayer import NeuronLayer

class NeuronNetwork:

	def __init__(self):
		self.num_inputs = []
		self.output = []	
		self.Neuron_layers = []	
		self.layers = []

	def create_layers(self, num_inputs: int, n_hidden_layers: list, num_outputs: int):
		"""
		Create a NeuronLayer for every layer (input layer + all hidden layers) and save it in a list.
		Then create the necessary Neurons in those layers.
		"""
		# Create a NeuronLayer for the initial input layer, plus additional NeuronLayers
		# for each hidden layer.
		self.Neuron_layers = [NeuronLayer() for _ in range(1 + len(n_hidden_layers))]
		self.layers = [num_inputs] + n_hidden_layers + [num_outputs]
		self.num_inputs = num_inputs

		for i in range(len(self.Neuron_layers)):
			self.Neuron_layers[i].create_Neurons(self.layers[i], self.layers[i+1])

	def set_weight(self, weight: list):
		"""
		Set the weights.
		Give each layer it's respective weights.
		"""
		if type(weight[0]) == list: # A nested list here means more than one layer.
			for counter, w in enumerate(weight):
				self.Neuron_layers[counter].set_weight(w)
		else:
			self.Neuron_layers[0].set_weight(weight)

	def set_bias(self, bias: list):
		"""
		Set the bias.
		Give each layer it's respective biases.
		"""
		if type(bias[0]) == list:
			for counter, b in enumerate (bias):
				self.Neuron_layers[counter].set_bias(b)
		else:
			self.Neuron_layers[0].set_bias(bias)

	def activate(self, input: list):
		"""
		Activation function.
		Feed the first layer with the initial input, then feed each following layer
		with the output from the layer before it. Rinse and repeat until there is a final output.
		"""
		next_input = input
		for i in range(len(self.Neuron_layers)):
			next_input = self.Neuron_layers[i].activate(next_input)
		self.output = next_input
		return self.output

	def __str__(self, input: list):
		"""
		Prints the input combination and the output result.
		"""
		for counter, i in enumerate(self.output):
			self.output[counter] = 1 if i >= 0.5 else 0

		print(f"Input {input} returns: {self.output}")