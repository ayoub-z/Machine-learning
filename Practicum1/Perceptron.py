class Perceptron: 

	def __init__(self):
		self.weight = []
		self.bias = None
		self.sum = 0	

	def set_weight(self, weights: list):
		"""
		Set the weight.
		"""		
		self.weight = weights

	def set_bias(self, bias: list):
		"""
		Set the bias.
		"""
		self.bias = bias

	def activate(self, input: list):
		"""
		Activation function. 
		Takes a binary input, such as [0, 1] and matches them with their corresponding 
		weights (in order). The inputs coupled with their weights is then summarized.
		"""
		self.sum = 0
		for counter, weight in enumerate(self.weight):
			self.sum += input[counter] * weight
		return 1 if self.sum - self.bias >= 0 else 0

	def __str__(self, input: list):
		"""
		Prints the input combination and the output result.
		"""
		print(f"Input {input} returns: {self.activate(input)}")