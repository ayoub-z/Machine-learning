from Perceptron import Perceptron

class PerceptronLayer:

	def __init__(self):
		self.perceptrons = []
		self.num_outputs = None
		self.output = []
		self.bias = []
		self.weights = []

	def create_perceptrons(self, num_inputs, num_outputs):
		"""
		Create n(num_inputs) number of Perceptrons(neurons) and save them as objects in a list.
		"""		
		self.perceptrons = [Perceptron() for _ in range(num_inputs)]
		self.num_outputs = num_outputs

	def set_weight(self, weights):
		"""
		Set the weight(s).
		"""
		for counter, weight in enumerate(weights):
			self.perceptrons[counter].set_weight(weight)

	def set_bias(self, bias):
		"""
		Set the bias.
		"""
		self.bias = bias

	def activate(self, input):
		"""
		Activation function.
		Loop through every Perceptron(neuron) and match each input with it's respective weight.
		Multiply the inputs with their weights and summarize them at the corresponding output index.
		At each output, if the total sum minus it's bias is greater than 0, the output is 1, else 0.
		"""			
		self.output = []
		sum = [0 for _ in range(self.num_outputs)]

		for i in range(len(self.perceptrons)): # For every neuron
			if self.num_outputs > 1: # If there is more than 1 output neuron
				# then loop through the nested list containing the weights
				for counter, weight in enumerate(self.perceptrons[i].weight):		
					sum[counter] += input[i] * weight
			else: # If there's only a single output neuron, add the
				  # we we can't loop through a nested list, since there's only one list
				sum[0] += input[i] * self.perceptrons[i].weight
		for i in range(self.num_outputs):
			if sum[i] - self.bias[i] >= 0:
				self.output.append(1)
			else:
				self.output.append(0)
		return self.output

	def __str__(self, input):
		"""
		Prints the input combination and the output result.
		"""
		print(f"Input {input} returns: {self.output}")		