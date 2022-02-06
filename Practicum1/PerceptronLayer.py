from Perceptron import Perceptron

class PerceptronLayer:

	def __init__(self):
		self.perceptrons = []
		self.num_inputs = None
		self.output = []
		self.bias = []
		self.weights = []

	def create_perceptrons(self, num_inputs, num_outputs):
		"""
		Create n number of Perceptrons and save them as objects in a list.
		"""		
		self.perceptrons = [Perceptron() for _ in range(num_outputs)]
		self.num_inputs = num_inputs

	def set_weight(self, weights):
		"""
		Set the weight.
		Each Perceptron(output), gets a list of weights that correspond with itself.
		Example: Consider the following weights: [[1, -1], [1, -1]] in a single layer.
		The weights leading to output 0 here are: [1, 1] and the weights leading to output 1 are [-1, -1]
		"""
		if len(self.perceptrons) > 1:
			for i in range(len(self.perceptrons)):
				temp_weight = []
				for weight in (weights):
					temp_weight.append(weight[i])
				self.perceptrons[i].set_weight(temp_weight)
		else:
			self.perceptrons[0].set_weight(weights)

	def set_bias(self, bias):
		"""
		Set the bias.
		Each Perceptron(output) gets it's own bias.
		"""
		for i in range(len(self.perceptrons)):
			self.perceptrons[i].bias = bias[i]
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
		for i in range(len(self.perceptrons)): # for each output
			for j in range(self.num_inputs): # for each input
				if len(self.perceptrons) > 1:
					weight = self.perceptrons[i].weight[j]
					if len(self.perceptrons) > 1:
						sum[i] += input[j] * weight
					else:
						sum[0] += input[j] * weight
				else:
					sum[0] += input[j] * self.perceptrons[0].weight[j]
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