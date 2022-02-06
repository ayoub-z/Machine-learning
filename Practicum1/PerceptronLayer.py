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

	def activate(self, input: list):
		"""
		Activation function.
		Loop through every Perceptron(output) and match each input with it's weight.
		Then multiply each input with it's respective weight and summarize them at the corresponding output(s).
		At each output, if the total sum minus it's bias is greater than 0, the output is a 1, else a 0.
		"""
		self.output = []
		sum = [0 for _ in range(len(self.perceptrons))]

		for i in range(len(self.perceptrons)): # For each output
			for j in range(self.num_inputs): # For each input
				if len(self.perceptrons) > 1:
					# At neuron 0 for example, it's weight to each output(Perceptron) is saved
					# on those outputs at index 0. Similarly for neuron 1 it's index 1, etc.
					weight = self.perceptrons[i].weight[j]
					sum[i] += input[j] * weight
				else:
					sum[0] += input[j] * self.perceptrons[0].weight[j]
		for i in range(len(self.perceptrons)):
			if sum[i] - self.perceptrons[i].bias >= 0:
				self.output.append(1)
			else:
				self.output.append(0)
		return self.output

	def __str__(self, input: list):
		"""
		Prints the input combination and the output result.
		"""
		print(f"Input {input} returns: {self.output}")