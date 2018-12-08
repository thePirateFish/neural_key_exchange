import torch
from tpm import TreeParityMachine
#import keras
#from keras.models import Sequential
#from keras.layers import LocallyConnected1D, Reshape
import numpy as np

i = 3

# K = number of hidden layer nodes
# N = number of inputs per hidden node
# L = [-L, +L] range each weight may take

k = 3
n = 4
l = 3
learning_rate = 1.0
tpm_alice = TreeParityMachine(k, n, l, learning_rate)
tpm_bob = TreeParityMachine(k, n, l, learning_rate)
# model = Sequential()
# model.add(Reshape(input_shape=(12,), target_shape=(12,1)))
# model.add(LocallyConnected1D(filters=3, kernel_size=n, strides=n,))

# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

torch.manual_seed(2)
while(i > 0):
	# 1 x k*n tensor of random ints, either -1, 0, or 1
	#rand_array = np.random.randint(0, high=2, size=(1, k*n))
	#rand_array[rand_array==0] = -1
	#print(rand_array)
	#rand_input = torch.tensor(rand_array)
	
	r_input = torch.randint(-1, 2, (1, k*n))
	r_input[r_input==0] = -1

	# #print(rand_input)
	# tpm_alice.output()
	# tpm_bob.output()
	result_a = tpm_alice(r_input)
	result_b = tpm_bob(r_input)
	# print(result_a)
	# print(result_b)
	if (result_a == result_b):
		print("MATCH")
	# else:
	# 	print("NO MATCH")
	# print("\n")
	i-=1

# rand_input = np.random.randint(-1, 2, size=k*n)
# rand_output = np.random.randint(-1, 2, size=1)

# x_val = np.random.randint(-1, 2, size=k*n)
# y_val = np.random.randint(-1, 2, size=1).reshape(1, 1)
# print(rand_input.shape)
# model.fit(rand_input, rand_output, batch_size=1, epochs=1, validation_data=(x_val, y_val))
