import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np
from torch.optim.optimizer import Optimizer, required


# k = 3
# n = 4
# l = 3


class TreeParityMachine(nn.Module):

	# num_hidden_layer_nodes = number of hidden layer nodes
	# num_inputs_per_node = number of inputs per hidden node
	# L = [-L, +L] range each weight may tanum_hidden_layer_nodese
	def __init__(self, k=3, n=4, l=3):
		super(TreeParityMachine, self).__init__()

		self.num_hidden_layer_nodes = k
		self.num_inputs_per_node = n
		self.weight_range = l
		self.update_rules = ["hebbian", "anti-hebbian", "random walk"]


		self.register_parameter("weights", nn.Parameter(torch.randint(-1*self.weight_range, self.weight_range+1, (1, self.num_hidden_layer_nodes*self.num_inputs_per_node)).float(), requires_grad=False))
		self.register_parameter("sigmas", nn.Parameter(torch.zeros([self.num_hidden_layer_nodes]), requires_grad=False))
		self.register_parameter("tau", nn.Parameter(torch.zeros([1]), requires_grad=False))
		self.last_input = torch.empty([1, self.num_hidden_layer_nodes*self.num_inputs_per_node])
		

	def forward(self, x):

		# tau = output of hidden layer
		self.last_input = x
		y = torch.chunk(x, self.num_hidden_layer_nodes, dim=1)
		#print(type(y))
		parts = list(y)

		weight_chunks = torch.chunk(self.weights, self.num_hidden_layer_nodes, dim=1)

		for i, weight_chunk in enumerate(weight_chunks):
			self.sigmas[i] = weight_chunk.mm(parts[i].transpose(0,1).float()).sign_()
		self.sigmas[self.sigmas==0] = -1

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
			raise ValueError("Update rule must be either hebbian, anti-hebbian, or random walk.")

		update_factor = 0
		for i, sigma in enumerate(self.sigmas):
			for j in range(self.num_inputs_per_node):
				index = i*self.num_inputs_per_node + j
			

				# Activation: we only update when sigma = tau = other_tau
				activation = theta(sigma.item(), self.tau.item()) * theta(self.tau.item(), other_tau)
				if (activation == 0):
					return

				if (rule == "hebbian"):
					update_factor = self.last_input[0][index] * self.tau.item() * activation
					self.weights[0][index] += update_factor
				elif (rule == "anti-hebbian"):
					update_factor = self.last_input[0][index] * self.tau.item() * activation
					self.weights[0][index] -= update_factor
				elif (rule == "random walk"):
					update_factor = self.last_input[0][index] * activation
					self.weights[0][index] += update_factor

				
		self.weights.clamp_(min=(-1*self.weight_range), max=(self.weight_range))

	def get_weights(self):
		for name, param in self.named_parameters():
			if name in ['weights']:
				return param


	# TESTING
	# intended for self-generating input
	def output(self):
		#np.random.seed(seed=2)
		rand_array = np.random.randint(0, high=2, size=(1, self.num_hidden_layer_nodes*self.num_inputs_per_node))
		rand_array[rand_array==0] = -1
		#print(rand_array)
		r_input = torch.tensor(rand_array)
		#print(np.random.rand(4))
		# r_input = torch.randint(0, 2, (1, self.num_hidden_layer_nodes*self.num_inputs_per_node))
		# r_input[r_input==0] = -1
		#print(r_input)
		self.forward(r_input)
		#print(self.forward(r_input))

	def initialize_seed(self, seed):
		#torch.manual_seed(seed)
		np.random.seed(seed=2)
	# def update(self, partner_tau, update_rule):

	# def update_hebbian(self, partner_tau):

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

def theta(a, b):
	if (a==b):
		return 1
	else:
		return 0



# define loss function
# define optimizer

