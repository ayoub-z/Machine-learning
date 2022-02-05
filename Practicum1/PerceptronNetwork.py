from PerceptronLayer import PerceptronLayer

class PerceptronNetwork:

	def __init__(self):
		self.num_inputs = []
		self.num_outputs = []
		self.output = []	
		self.perceptron_layers = []	
		self.layers = []

	def create_layers(self, num_inputs, n_hidden_layers, num_outputs):
		self.perceptron_layers = [PerceptronLayer() for _ in range(1 + len(n_hidden_layers))]
		self.layers = [num_inputs] + n_hidden_layers + [num_outputs]
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs

		for i in range(len(self.perceptron_layers)):
			self.perceptron_layers[i].create_perceptrons(self.layers[i], self.layers[i+1])

	def set_weight(self, weights):
		if type(weights[0]) == list:
			for counter, weight in enumerate(weights):
				self.perceptron_layers[counter].set_weight(weight)
		else:
			self.perceptron_layers[0].set_weight(weights)

	def set_bias(self, bias):
		if type(bias[0]) == list:
			for counter, b in enumerate (bias):
				self.perceptron_layers[counter].set_bias(b)
		else:
			self.perceptron_layers[0].set_bias(bias)


	def activate(self, input):
		next_input = input
		for i in range(len(self.perceptron_layers)):
			next_input = self.perceptron_layers[i].activate(next_input)
		self.output = next_input
		return self.output

	def __str__(self, input):
		"""
		Prints the input combination and the output result.
		"""
		print(f"Input {input} returns: {self.output}")