from NeuronLayer import NeuronLayer
import decimal

class NeuronNetwork:

	def __init__(self):
		self.output = []	
		self.neuron_layers = []	
		self.layers = []
		self.error = []

	def create_layers(self, num_inputs: int, hidden_layers: list, num_outputs: int):
		"""
		Create a NeuronLayer for every layer (all hidden layers + output layer) and save it in a list.
		Then create the necessary neurons in those layers.
		"""

		# Create layers
		self.neuron_layers = [NeuronLayer() for _ in range(1 + len(hidden_layers))]
		self.layers = [num_inputs] + hidden_layers + [num_outputs]

		# Create neurons in each layer
		for i in range(len(self.neuron_layers)):
			self.neuron_layers[i].create_neurons(self.layers[i], self.layers[i+1])

	def get_error(self, target, output):
		return output * (1 - output) * -(target - output)

	def feed_forward(self, input, testing=False):
		"""
		Activate each neuron in the network starting from the initial layer, all
		the way until the output layer. Then return the output of the output layer.
		"""		
		next_input = input
		for i in range(len(self.neuron_layers)):
			next_input = self.neuron_layers[i].feed_forward(next_input)

		self.output = next_input

		output = self.output
		if testing: # If we're testing, we want the values to either be a 0 or 1
			for counter, i in enumerate(output):
				output[counter] = 1 if i >= 0.5 else 0																																					
			return output	
		
		return self.output

	def backpropagation(self, target, input, l_rate):
		"""
		Backpropagation function.
		After activating the entire network, backwords propagate starting from the output layer
		and continuing backwards to the hidden layers."""
		# Wrap target into list if it's a single number. Example:
		# if target is 1, wrap it into [1]. This is so we can loop over it
		if type(target) != list:
			target = [target]

		next_weights = []
		self.error = [] # reset error

		# Back propagate output layer
		for index, neuron in enumerate(self.neuron_layers[-1].neurons): # for neurons in last layer
			try:
				previous_output = self.neuron_layers[-2].output # output from HIDDEN layer before output layer
			except: # If there is no hidden layer, previous output is the original input
				previous_output = input
			self.error.append(neuron.get_error(neuron.output, target[index])) # save error to current output layer

			neuron.get_delta_weight(l_rate, previous_output) # Save delta weights to their neurons
			neuron.get_delta_bias(l_rate) # Save delta bias to it's neurons
			next_weights.append(neuron.weight) # Save current weights for the next layer

		next_errors = self.error # Save error so next layer can use it
		previous_output = [] # reset previous output

		# Back propagate hidden layer(s)
		for layer_count, layer in enumerate(self.neuron_layers[-2::-1]): # Iterate through hidden layers backwards
			# "self.neuron_layers[-2::-1]" contains ALL HIDDEN layers, backwards
			# when we add [layer_count + 1].output to the back of that, we say: 
			# give me the output of the hidden layer after the current one we're on
			try:
				privous_output = self.neuron_layers[-2::-1][layer_count + 1].output
			# In the case that there are no more hidden layers, it means we want the initial layer
			# And the output(s) of the initial neuron(s), is simply the input we give at the start
			except: 
				previous_output = input # The output of the initial neurons is the input in that case
			
			new_errors = layer.save_error(next_errors, next_weights) # Error hidden layer(s)
			next_errors = new_errors # Save error for next layer

			layer.get_delta_weight(l_rate, previous_output, layer.error) # Save delta weights to their neurons
			layer.get_delta_bias(l_rate)# Save delta bias to it's neurons

			next_weights = [] # Save the weights of current layer so next layer can use them
			for neuron in layer.neurons:
				next_weights.append(neuron.weight)

	def update(self):
		"""
		Update all weights and biases in each neuron by subtracting their delta values.
		"""		
		for layer in self.neuron_layers:
			for neuron in layer.neurons:
				neuron.update()

	def get_weights_bias(self):
		"""
		This function is simply used for TESTING. It prints out the values of the weights
		and bias of each neuron when done training.
		"""
		for layer in self.neuron_layers:
			for neuron in layer.neurons:
				print("\nFinal weights ", neuron.weight, sum(neuron.weight))
				print("Final bias ", neuron.bias)


	def total_cost(self, outputs, target):
		"""
		Calculate the total cost of all iterations.
		"""
		if len(self.output) > 1: # If there are multiple output neurons
			cost = 0
			for n, output in enumerate(outputs): # Calculate cost for each neuron and add up
				for counter, o in enumerate(output):
					cost += (target[n][counter] - o) ** 2
			return cost / (2 * len(outputs)) # Then divide them by 2 and the amount of iterations
		else:
			temp_sum = 0
			for counter, o in enumerate(outputs):
				temp_sum += (target[counter] - o) ** 2
			return temp_sum / (2 * len(outputs))


	def train(self, data: list, target_data, l_rate, epochs):
		"""
		Train function.
		Activate all the neurons (feed_forward). Then backwards propagate and update
		all the weights and biases. Rinse and repeat for each iteration. Rinse and 
		repeat for each epoch.
		"""
		for epoch in range(1, epochs + 1):
			outputs = [] # Used as parameter for cost function in proper format
			self.output = []

			for counter, input in enumerate(data): # loop over all iterations
				target = target_data[counter]
				output = self.feed_forward(input)
				
				# Unwrap output if there's only one
				if len(output) == 1:
					output = output[0]
				outputs.append(output)
				
				self.backpropagation(target, input, l_rate)
				self.update()

			total_cost = self.total_cost(outputs, target_data)
			if total_cost < 0.01:
				print("Total cost is below 0.01. Ending training early at epoch:", epoch)
				return
		print(f"Training finished after: {epoch} epochs\nTotal cost: {total_cost}")

	def __str__(self, input: list, testing=False):
		"""
		Prints the input combination and the output result.
		"""
		output = []
		for inp in input:
			output = self.feed_forward(inp, testing=True)
			print(f"Input {inp} returns: {output}")
		self.get_weights_bias()