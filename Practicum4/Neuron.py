import random

class Neuron: 

	def __init__(self):
		self.weight = []
		self.bias = None
		self.sum = 0
		self.outputs_list = []
		self.MSE = None
		self.succesful_iterations = 0
		self.num_inputs = 0

	def randomize_weights_bias(self):
		"""
		Set random weights/bias.
		"""
		for _ in range(self.num_inputs):
			self.weight.append(round(random.uniform(-1, 1), 2))
		self.bias = round(random.uniform(-1, 1), 2)

	def get_output(self, input: list):
		"""
		Return output from a given input.
		"""
		self.sum = 0

		for counter, weight in enumerate(self.weight):
			self.sum += input[counter] * weight
		
		return 1 if self.sum + self.bias >= 0 else 0

	def loss(self, d,y):
		"""
		Returns the loss.
		Hinge loss function: https://en.wikipedia.org/wiki/Hinge_loss
		"""
		return max(0,1-d*y)

	def activate(self, data, target, epochs, l_rate, verbose):
		"""
		Activation function.
		Takes a list containing all of the inputs, a list with all the targets, 
		the amount of epochs to run, the learning rate and the verbose to print out the process.
		Then we run the update function for every epoch or until we get the desired output on each iteration.
		"""
		self.num_inputs = len(data[0])
		self.randomize_weights_bias()

		for epoch in range(epochs):
			for counter, i in enumerate(data): # for every possible input
				input = i
				self.update(l_rate, target[counter], epoch, counter, input, verbose)
			if self.succesful_iterations >= len(data): # stop if we were succesful in all iterations
				print(f"\nModel is 100% succesful after {epoch+1} epochs. Stopping...")
				print(f"Final bias = {self.bias} and weights are {self.weight}")
				return self.get_output(input)
		print(f"\nModel is unable to perfrom succesfully 100% of the time. Stopping after {epoch+1} epochs...")
		print(f"Final bias = {self.bias} and weights are {self.weight}\n")
		return self.get_output(input)

	def update(self, l_rate, d, epoch, counter, input, verbose):
		"""
		Update function.
		Run the perceptron and see if it's the the desired output.
		If so, we stop. Otherwise we adjust the weights/bias accordingly.
		"""
		starting_output = self.get_output(input)
		if starting_output == d: # Stop if output is already correct
			self.succesful_iterations += 1
			return starting_output
		else:
			self.succesful_iterations = 0
			output = self.get_output(input)

			for counter, weight in enumerate(self.weight): # for every input neuron
				if input[counter] > 0: # only activate the weights if their input is above 0
					self.weight[counter] += (l_rate * (d-output))
					self.weight[counter] = round(self.weight[counter], 2)  # rounding to 2 decimals

			self.bias += (l_rate * (d-output))
			self.bias = round(self.bias, 2) # rounding to 2 decimals

			if verbose == True:
				print(f"\nInput: {input} gives output {starting_output}. While desired output is {d}")
				print(f"Starting weights are: {self.weight}\nStarting bias is: {self.bias}\n")			
				print(f"After applying learning rule, the new output is {self.get_output(input)} / expected output is {d}. \nThe updated weights are: {self.weight} and bias is: {self.bias}\n")

	def __str__(self, input, epochs, l_rate, answer):
		"""
		Prints the input combination and the output result.
		"""
		print(f"Input {input} returns: {self.activate(input, epochs, l_rate, answer)}")