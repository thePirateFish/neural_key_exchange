# Author: Matthew Gigliotti
# Fall 2018
# 
# reference: credit to https://github.com/farizrahman4u/neuralkey
# for calculating synchronizations

import torch
from tpm import TreeParityMachine
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt

i = 0
matches = 0
percent_synced = 0
sync_list = []

# Default settings
# K = number of hidden layer nodes
# N = number of inputs per hidden node
# L = [-L, +L] range each weight may take
k = 3
n = 4
l = 3

update_rule = "hebbian"
#update_rule = "anti-hebbian"
#update_rule = "random walk"
#update_rules = ["hebbian", "anti-hebbian", "random walk"]

def calculate_percent_synced(wa, wb):
	result = ((1.0 - torch.mean(1.0 * torch.abs(wa - wb) / (2*l))).item()) * 100
	return round(result, 3)

def generate_random_input():
	r_input = torch.randint(-1, 2, (1, k*n))
	r_input[r_input==0] = -1
	return r_input

tpm_alice = TreeParityMachine(k, n, l)
tpm_bob = TreeParityMachine(k, n, l)

#torch.manual_seed(2)
start = timer()
while(percent_synced != 100):
	r_input = generate_random_input()
	tau_a = tpm_alice(r_input)
	tau_b = tpm_bob(r_input)

	if (tau_a == tau_b):
		tpm_alice.update(tau_b, rule=update_rule)
		tpm_bob.update(tau_a, rule=update_rule)
		wa = tpm_alice.get_weights()
		wb = tpm_bob.get_weights()
		percent_synced = calculate_percent_synced(wa, wb)
		matches+=1

	sync_list.append(percent_synced)

	if (i % 10 == 0 or percent_synced == 1):
		print("Percent synced: %.2f" % percent_synced)

	i+=1

end = timer()
print("Number of iterations: %d" % i)
print("Elapsed time: %.2f" % (end-start))
print("Matches: %d" % matches)
print("Alice's final weights: %s" % wa)
print("Bob's final weights: %s" % wb)
#print(sync_list)
style = "Solarize_Light2"
with plt.style.context(style):
	timesteps = np.arange(0, i, 1)
	fig, ax = plt.subplots(num=style)
	ax.plot(timesteps, sync_list)
	ax.set(xlabel="Learning Steps", ylabel="% Synchronized", title="Synchronization Example Using %s Learning" % update_rule.title())

plt.show()

