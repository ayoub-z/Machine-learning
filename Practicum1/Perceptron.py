class Perceptron: 

	def __init__(self):
		self.weights = {}
		self.bias = None

	def set_weight(self, x, n, x2):
		weight = {}
		weight[x2] = n
		self.weights[x] = weight
		
	def set_bias(self, n):
		self.bias = n

	def activate(self, input):

		sum = 0
		for counter, i in enumerate(input):
			if i == 1:

				weight = list(list(self.weights.values())[counter].values())[0]
				sum += weight * 1

		self.__str__(sum, input)

	def __str__(self, sum, input):
		if sum >= self.bias:
			print(f"Combination {input} returns True")
		else:
			print(f"Combination {input} returns False")

