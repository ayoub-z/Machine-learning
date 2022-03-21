import random
import math
import decimal

class Neuron: 

	def __init__(self):
		self.weight = []
		self.bias = None
		self.error = None
		self.output = None
		self.delta_weights = []
		self.delta_bias = None

	def randomize_weights_bias(self, num_inputs):
		"""
		Set random weights/bias.
		"""
		for _ in range(num_inputs):
			self.weight.append(round(random.uniform(-1, 1), 2))
		self.bias = round(random.uniform(-1, 1), 2)

	def get_error(self, output, desired_output):
		"""
		This function is used to return the error of the OUTPUT neuron.
		"""
		self.error = output * (1 - output) * -(desired_output - output)
		return self.error

	def get_delta_weight(self, l_rate, previous_output):
		"""
		This function is used to return the delta of each weight in this output neuron.
		"""		
		self.delta_weights = []

		for counter, weight in enumerate(self.weight):
			self.delta_weights.append(l_rate * previous_output[counter] * self.error)
		return self.delta_weights

	def get_delta_weight_hidden(self, l_rate, previous_output, error):
		"""
		This function is used to return the delta of each weight in the HIDDEN layer.
		Since calculating the delta of the weights in the hidden layer requires a different
		calculation than that of the output layer, we need a different function for it.
		"""				
		self.delta_weights = []

		for output in previous_output:
			self.delta_weights.append(l_rate * output * error)
		return self.delta_weights

	def get_delta_bias(self, l_rate):
		"""
		Calculate the delta of the bias(es).
		"""
		self.delta_bias =  l_rate * 1 * self.error
		return self.delta_bias

	def sigmoid(self, n):
		"""
		Sigmoid activation function.
		"""
		return 1 / (1 + math.e**-n)


	def feed_forward(self, input):
		"""
		Activate the neuron and return it's output.
		"""
		sum = 0
		for counter, i in enumerate(input):				
			sum += (i * self.weight[counter])	

		self.output = self.sigmoid(sum + self.bias)
		return self.output

	def update(self):
		"""
		Update all weights and the bias by subtracting their delta values.
		"""
		for counter, delta_weight in enumerate(self.delta_weights):
			self.weight[counter] = self.weight[counter] - delta_weight

		self.bias = self.bias - self.delta_bias

	def __str__(self, input):
		"""
		Prints the input combination and the output result.
		"""
		print(f"Input {input} returns: {self.output}")