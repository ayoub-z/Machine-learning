from Neuron import Neuron

class NeuronLayer:

	def __init__(self):
		self.neurons = []
		self.num_inputs = None
		self.output = []
		self.error = []

	def create_neurons(self, num_inputs: int, num_outputs: int):
		"""
		Create a Neuron for every output and save it in a list.
		"""		
		self.neurons = [Neuron() for _ in range(num_outputs)]
		self.num_inputs = num_inputs

		for neuron in self.neurons:
			neuron.randomize_weights_bias(num_inputs)

	def get_delta_weight(self, l_rate, previous_output, error):
		"""
		This function is used to return the delta of each weight in this HIDDEN layer.
		The function "get_delta_weight_hidden" inside this layer's neurons is further 
		used to calculate the delta values of the weights of those neurons.
		"""			
		delta_weights = []

		for counter, neuron in enumerate(self.neurons):
			delta_weights.append(neuron.get_delta_weight_hidden(l_rate, previous_output, error[counter]))
		return delta_weights

	def get_delta_bias(self, l_rate):
		"""
		This function is used to return the delta of each bias in this HIDDEN layer.
		The function "get_delta_bias" inside this layer's neurons is further used to 
		calculate the delta values of the weights of those neurons.
		"""		
		error = []
		for counter, neuron in enumerate(self.neurons):
			error.append(neuron.get_delta_bias(l_rate))
		return error

	def save_error(self, next_layer_errors, next_layer_weights):
		"""
		Calculate the error of each neuron in a HIDDEN LAYER and saves them
		locally to each neuron.
		"""
		self.error = []
		for index2, output in enumerate(self.output):
			# For each neuron we're currently on, save the weights that are linked to it.
			next_weights = [next_layer_weights[0][index2] for _ in range(len(next_layer_errors))]

			sum = 0
			for index, error in enumerate(next_layer_errors):
				sum += error * next_weights[index]
			self.error.append(output * (1 - output) * sum)

		for counter, neuron in enumerate(self.neurons):
			neuron.error = self.error[counter] # save errors to each neuron locally
		return self.error

	def feed_forward(self, input):
		"""
		Activate each neuron in this layer and return the output(s) of the neurons.
		"""		
		self.output = [] # clear output cache

		for i in range(len(self.neurons)):
			self.output.append(self.neurons[i].feed_forward(input))
		return self.output

	def __str__(self, input: list):
		"""
		Prints the input combination and the output result.
		"""
		for counter, i in enumerate(self.output):
			self.output[counter] = 1 if i >= 0.5 else 0
		print(f"Input {input} returns: {self.output}")