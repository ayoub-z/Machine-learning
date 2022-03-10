import math

class Neuron: 

	def __init__(self):
		self.weight = []
		self.bias = None
		self.sum = 0

	def set_weight(self, weight: list):
		"""
		Set the weight.
		"""
		self.weight = weight

	def set_bias(self, bias: int):
		"""
		Set the bias.
		"""
		self.bias = bias

	def sigmoid(self, n):
		"""
		Sigmoid activation function.
		"""
		return 1 / (1 + math.e**-n)

	def activate(self, input: list):
		"""
		Activation function.
		Takes a binary input, such as [0, 1] and matches them with their corresponding
		weights (in the given order). The inputs coupled with their weights is then summarized.
		If it's the LAST LAYER, a 1 is returned if sum is bigger than bias, other a 0.
		If it's not the last layer, we simply return the output of the sigmoid function.
		"""
		self.sum = 0

		for counter, weight in enumerate(self.weight):
			self.sum += input[counter] * weight	

		output = self.sigmoid(self.sum + self.bias)
		return output

	def __str__(self, input: list):
		"""
		Prints the input combination and the output result.
		"""
		output = 1 if self.activate(input) >= 0.5 else 0
		print(f"Input {input} returns: {output}")