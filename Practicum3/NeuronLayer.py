from Neuron import Neuron

class NeuronLayer:

	def __init__(self):
		self.Neurons = []
		self.num_inputs = None
		self.output = []

	def create_Neurons(self, num_inputs: int, num_outputs: int):
		"""
		Create a Neuron for every output and save it in a list.
		"""		
		self.Neurons = [Neuron() for _ in range(num_outputs)]
		self.num_inputs = num_inputs

	def set_weight(self, weight: list):
		"""
		Set the weight.
		Each Neuron(output), gets a list containing the weights of each neuron that's linked to itself.
		Example: Consider the following weights: [[1, -1], [1, -1]] in a single layer with 2 neurons/ 2 outputs.
		The weights leading to output 0 here are: [1, 1] and the weights leading to output 1 are [-1, -1].
		"""
		if len(self.Neurons) > 1:
			for i in range(len(self.Neurons)):
				temp_weight = []
				for w in (weight):
					temp_weight.append(w[i])
				self.Neurons[i].set_weight(temp_weight)
		else:
			self.Neurons[0].set_weight(weight)

	def set_bias(self, bias: list):
		"""
		Set the bias.
		Each Neuron(output) has it's own bias.
		"""
		for i in range(len(self.Neurons)):
			self.Neurons[i].bias = bias[i]

	def activate(self, input: list):
		"""
		Activation function.
		Call the activation function from the Neuron for each output neuron.
		We append that output to the output list and finally return it once we're done.
		"""
		self.output = []

		for i in range(len(self.Neurons)):
			self.output.append(self.Neurons[i].activate(input))
		return self.output

	def __str__(self, input: list):
		"""
		Prints the input combination and the output result.
		"""
		for counter, i in enumerate(self.output):
			self.output[counter] = 1 if i >= 0.5 else 0

		print(f"Input {input} returns: {self.output}")