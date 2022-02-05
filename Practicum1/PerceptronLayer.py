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
		Create n number of Perceptrons and save them as objects in a list.
		"""		
		self.perceptrons = [Perceptron() for _ in range(num_inputs)]
		self.num_outputs = num_outputs

	def set_weight(self, weights):
		"""
		Set the weight.
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
		Takes a binary input, such as [0, 1] and matches them with their corresponding 
		weights (in order). The inputs are multiplied with their weights and are then summarized 
		and compared with their respective bias(output neuron).
		This is done one by one for each weight in each neuron per Perceptron.
		"""		
		self.output = []
		sum = [0 for _ in range(self.num_outputs)]
		for i in range(len(self.perceptrons)):
			if type(self.perceptrons[i].weight) == list:
				for counter, weight in enumerate(self.perceptrons[i].weight):		
					if self.num_outputs > 1:
						sum[counter] += input[i] * weight
					else:
						sum[0] += input[i] * weight
			else:
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