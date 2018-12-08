import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TreeParityMachine(nn.Module):

	# num_hidden_layer_nodes = number of hidden layer nodes
	# num_inputs_per_node = number of inputs per hidden node
	# L = [-L, +L] range each weight may tanum_hidden_layer_nodese
	# lr = learning rate
	def __init__(self, num_hidden_layer_nodes=3, num_inputs_per_node=4, weight_range=6, lr=1.0):
		super(TreeParityMachine, self).__init__()
		#self.initialize_seed(2)

		self.num_hidden_layer_nodes = num_hidden_layer_nodes
		self.num_inputs_per_node = num_inputs_per_node
		self.weight_range = weight_range
		self.hidden = nn.ModuleList()
		self.sigmas = torch.zeros([1, 3], dtype=torch.int32)
		

		for num_hidden_layer_nodes in range(num_hidden_layer_nodes):
			self.hidden.append(nn.Linear(self.num_inputs_per_node, 1))
			torch.nn.init.uniform_(self.hidden[num_hidden_layer_nodes].weight, a=(-1)*self.weight_range, b=self.weight_range+1)

		#self.weights_layer = torch.nn.Linear(1, num_hidden_layer_nodes, num_inputs_per_node)
		# self.weights_layer = torch.nn.Conv1d(1, 1, kernel_size=self.n, stride=self.n, groups=self.num_hidden_layer_nodes)
		# torch.nn.init.constant_(self.weights_layer.weight, 1)

		# self.linears = nn.ModuleList()

		# for i in range(self.num_hidden_layer_nodes):
		# 	linears.append(torch.nn.Linear(self.n, 1))
		# for x, l in enumerate(self.linears):
		# 	torch.nn.init.constant_(l.weight, 1)

	def forward(self, x):

		# tau = output of hidden layer
		#for param in self.weights_layer.parameters():
		#	print(param.data)
		# print(self.weights_layer.weight)
		# print(x)
		# x = x.reshape(1, self.num_hidden_layer_nodes, self.n)
		# print(x)
		# x = self.weights_layer(x)
		# print(x)
		# x = x.sign()

		#x = torch.split(x, self.num_hidden_layer_nodes, dim=1)
		y = torch.chunk(x, self.num_hidden_layer_nodes, dim=1)
		#print(type(y))
		parts = list(y)
		#print(parts)
		for i, l in enumerate(self.hidden):
			#print(parts[i])
			parts[i] = l(parts[i])
			#print(parts[i])
			#parts[i] = parts[i].sign_()
			self.sigmas = parts[i].sign_()
			#print(parts[i])

		z = torch.tensor(self.sigmas)
		#print(z)
		tau = torch.prod(z)
		print(tau.item())
		#print("hello")
		return tau.item()

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




# define loss function
# define optimizer

