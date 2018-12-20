# Author: Matthew Gigliotti
# Fall 2018

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.optimizer import Optimizer, required

"""
An implementation of a Tree Parity Machine as a PyTorch nn.Module.
Default values:
	k = 3	->	The number of hidden layer nodes
	n = 4	->	Inputs per hidden node
	l = 3	->	Synaptic depth

An input vector must be a 1 x (k*n) tensor of 1's and -1's
The default input vector shape is 1 x 12, with 4 inputs to each hidden node.
Weights may take a range [-l, +l], and will be clamped to stay in that range.

"""
class TreeParityMachine(nn.Module):
    
	def __init__(self, k=3, n=4, l=3):
		super(TreeParityMachine, self).__init__()

		self.num_hidden_layer_nodes = k
		self.num_inputs_per_node = n
		self.weight_range = l
		self.update_rules = ["hebbian", "anti-hebbian",
		"random walk"]

		input_length = self.num_hidden_layer_nodes*self.num_inputs_per_node

		self.register_parameter("weights",
		nn.Parameter(torch.randint(-1*self.weight_range,
									self.weight_range+1,
							 		(1, input_length)).float(),
									requires_grad=False)) 

		self.register_parameter("sigmas",
		nn.Parameter(torch.zeros([self.num_hidden_layer_nodes]),
								requires_grad=False)) 

		self.register_parameter("tau",
		nn.Parameter(torch.zeros([1]), requires_grad=False)) 

		self.last_input = torch.empty([1, input_length])
		

	def forward(self, x):
		self.last_input = x
		y = torch.chunk(x, self.num_hidden_layer_nodes, dim=1)
		parts = list(y)
		weight_chunks = torch.chunk(self.weights,
									self.num_hidden_layer_nodes,
									dim=1)

		for i, weight_chunk in enumerate(weight_chunks):
			self.sigmas[i] = \
			weight_chunk.mm(parts[i].transpose(0,1).float()).sign_()
		self.sigmas[self.sigmas==0] = -1

		# tau = output of hidden layer
		self.tau[0] = torch.prod(self.sigmas)

		return self.tau.item()

	def print_parameters(self): 
		for name, param in self.named_parameters():
			print("Name: %s" % name)
			print(param.data)

	def print_layer(self, id, node): 
		print("Node %d" % id)
		print("\t%s" % node)
		print("\tWeights: %s" % node.weight.data)

	def update(self, other_tau=required, rule=required):
		if rule not in self.update_rules:
			raise ValueError(
		"Update rule must be either hebbian, anti-hebbian, or random walk.")

		update_factor = 0
		for i, sigma in enumerate(self.sigmas):
			for j in range(self.num_inputs_per_node):
				index = i*self.num_inputs_per_node + j
				activation = theta(sigma.item(), self.tau.item())* \
							 theta(self.tau.item(), other_tau)
				
				if (activation == 0):
					return

				if (rule == "hebbian"):
					update_factor = self.last_input[0][index]* \
					self.tau.item()*activation
					self.weights[0][index] += update_factor
				elif (rule == "anti-hebbian"):
					update_factor = self.last_input[0][index]* \
					self.tau.item()*activation
					self.weights[0][index] -= update_factor 
				elif (rule == "random walk"):
					update_factor = self.last_input[0][index]*activation
					self.weights[0][index] += update_factor

				
		self.weights.clamp_(min=(-1*self.weight_range),
							max=(self.weight_range))

	def get_weights(self):
		for name, param in self.named_parameters():
			if name in ['weights']:
				return param

""" Returns 1 if values are equal, 0 otherwise.
Used to determine the weight updates, based on the agreement
of both tau values, and the tau and sigma values.

"""
def theta(a, b):
    if (a==b):
    	return 1
    else:
    	return 0

""" Return a random input vector

Under development. Intended to be a standalone input generator
such that each party in the protocol can generate the same
random sequence individually, Given an agreed upon random seed.
This may not be possible.

"""   
# def output(self):
# np.random.seed(seed=2)         
# rand_array = np.random.randint(0, high=2,
# size=(1, self.num_hidden_layer_nodes*self.num_inputs_per_node))
# rand_array[rand_array==0] = -1         
# print(rand_array)         
# r_input = torch.tensor(rand_array)
# print(np.random.rand(4))         
# r_input = torch.randint(
#	0, 2, (1, self.num_hidden_layer_nodes*self.num_inputs_per_node))
# r_input[r_input==0] = -1
# print(r_input)
# self.forward(r_input)
# print(self.forward(r_input))

""" NOT USED
Sets the random seed. Used for setting the seed with numpy vs. PT.

"""
def initialize_seed(self, seed):
	np.random.seed(seed=2)

""" NOT USED

Under development.
A custom optimizer for a TPM performing synchronization. Determined to
not be worth using, since there is no real loss function. The updates
can be more easily performed manually using the update method.

"""
class TPM_Update(Optimizer):
	def __init__(self, params, other_tau=required, rule="hebbian"):
		defaults = dict(other_tau=other_tau, rule=rule)
		super(TPM_Update, self).__init__(params, defaults)

	def __setstate__(self, state):
		super(TPM_Update, self).__setstate__(state)
	
	def step(self):
		for group in self.param_groups:
			other_tau = group['other_tau']
			rule = group['rule']

			for p in group['params']:
				print(p)

