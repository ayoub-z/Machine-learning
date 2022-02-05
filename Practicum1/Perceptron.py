class Perceptron: 

	def __init__(self):
		self.weight = []
		self.bias = None		

	def set_weight(self, weights=list):
		"""
		Set the weight.
		"""		
		self.weight = weights

	def set_bias(self, bias):
		"""
		Set the bias.
		"""
		self.bias = bias

	def activate(self, input):
		"""
		Activation function. 
		Takes a binary input, such as [0, 1] and matches them with their corresponding 
		weights (in order). The inputs coupled with their weights is then summarized. 
		Finally the __str__ method is called with the result and input as the parameters.
		"""
		sum = 0
		for counter, i in enumerate(input):
			if i == 1:
				sum += self.weight[counter] * i
		return 1 if sum >= self.bias else 0

	def __str__(self, input):
		"""
		Prints the input combination and the output result.
		"""

		print(f"Input {input} returns: {self.activate(input)}")

# inputs = [[0, 1, 1], [0, 1, 0], [0, 0, 0]]
# weights = [0.2, 0.3, 0.4]