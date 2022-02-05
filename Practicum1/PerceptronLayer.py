from Perceptron import Perceptron

class PerceptronLayer:

	def __init__(self):
		self.perceptrons = []
		self.output = []

	def create_perceptrons(self, num_outputs):
		"""
		Create n number of Perceptrons and save them as objects in a list.
		"""		
		self.perceptrons = [Perceptron() for _ in range(num_outputs)]

	def set_weight(self, weights):
		"""
		Set the weight.
		"""
		if type(weights[0]) == list:
			for counter, weight in enumerate(weights):
				self.perceptrons[counter].set_weight(weight)
		else:
			self.perceptrons[0].set_weight(weights)

	def set_bias(self, biases):
		"""
		Set the bias.
		"""		
		for counter, bias in enumerate (biases):
			self.perceptrons[counter].set_bias(bias)

	def activate(self, input):
		"""
		Activation function. 
		Takes a binary input, such as [0, 1] and matches them with their corresponding 
		weights (in order). The inputs are multiplied with their weights and are then summarized 
		and compared with their respective bias(output neuron).
		This is done one by one for each weight in each neuron per Perceptron.
		"""		
		self.output = []
		sum = [0 for _ in range(len(self.perceptrons))]
		
		for i in range(len(self.perceptrons)):
			self.perceptrons[i].activate(input)
			for counter, weight in enumerate(self.perceptrons[i].weight):
				sum[counter] += input[counter] * weight

		for i in range(len(self.perceptrons)):
			if sum[i] - self.perceptrons[i].bias >= 0:
				self.output.append(1)
			else:
				self.output.append(0)

		return self.output

	def __str__(self, input):
		"""
		Prints the input combination and the output result.
		"""
		print(f"Input {input} returns: {self.output}")		