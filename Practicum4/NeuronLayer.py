from Perceptron import Perceptron

class PerceptronLayer:

	def __init__(self):
		self.perceptrons = []
		self.num_inputs = None
		self.output = []

	def create_perceptrons(self, num_inputs: int, num_outputs: int):
		"""
		Create a Perceptron for every output and save it in a list.
		"""		
		self.perceptrons = [Perceptron() for _ in range(num_outputs)]
		self.num_inputs = num_inputs

	def set_weight(self, weight: list):
		"""
		Set the weight.
		Each Perceptron(output), gets a list containing the weights of each neuron that's linked to itself.
		Example: Consider the following weights: [[1, -1], [1, -1]] in a single layer with 2 neurons/ 2 outputs.
		The weights leading to output 0 here are: [1, 1] and the weights leading to output 1 are [-1, -1].
		"""
		if len(self.perceptrons) > 1:
			for i in range(len(self.perceptrons)):
				temp_weight = []
				for w in (weight):
					temp_weight.append(w[i])
				self.perceptrons[i].set_weight(temp_weight)
		else:
			self.perceptrons[0].set_weight(weight)

	def set_bias(self, bias: list):
		"""
		Set the bias.
		Each Perceptron(output) has it's own bias.
		"""
		for i in range(len(self.perceptrons)):
			self.perceptrons[i].bias = bias[i]

	def activate(self, input: list, last_layer=True):
		"""
		Activation function.
		Call the activation function from the perceptron for each output neuron.
		We append that output to the output list and finally return it once we're done.
		"""
		self.output = []

		for i in range(len(self.perceptrons)):
			self.output.append(self.perceptrons[i].activate(input, last_layer))
		return self.output

	def __str__(self, input: list):
		"""
		Prints the input combination and the output result.
		"""
		print(f"Input {input} returns: {self.output}")