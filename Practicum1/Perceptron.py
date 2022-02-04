class Perceptron: 

	def __init__(self):
		self.weights = {}
		self.bias = None

	def set_weight(self, x, n, x2):
		"""
		x = initial neuron.
		n = the weight.
		x2 = neuron the weight will be linked to.

		Sets the weight in the form of a nested dict. Creates a dict with the 
		key being the initial neuron and the value being another dict containing
		the neuron it's linked to with the according weight.
		Example: {'X1':{'X3':0.5}}
		"""
		weight = {}
		weight[x2] = n
		self.weights[x] = weight
		
	def set_bias(self, n):
		"""
		Sets bias.
		"""
		self.bias = n

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
				# For example, it turns the dict {'X1':{'O':0.5}} into 
				# [{'O':0.5}][counter] and then further into [0.5][0]
				weight = list(list(self.weights.values())[counter].values())[0]
				sum += weight * 1

		self.__str__(sum, input)

	def __str__(self, sum, input):
		"""
		Prints the input combination with a "success" string if the result
		is bigger than or equal to the bias. Otherwise prints out the input 
		combination with a "failure" string.
		"""
		if sum >= self.bias:
			print(f"Combination {input} returns True")
		else:
			print(f"Combination {input} returns False")

