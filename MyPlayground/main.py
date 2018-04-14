# -*- coding: utf-8 -*- 
#!/usr/bin/env python
'''
Student name: Beier (Benjamin) Liu
Date: 

Remark:
Python 2.7 is recommended
Before running please install packages *numpy
Using cmd line py -2.7 -m install [package_name]
'''
import os, time, logging
import copy, math
import functools, itertools
import numpy as np 
from Implementations.initialize_parameters import *
from Implementations.L_model_forward import *
from Implementations.L_model_backward import *
from Implementations.compute_cost import *
from Implementations.update_parameters import *
logging.getLogger().setLevel(logging.INFO)


'''===================================================================================================
Main program:
Create shallow and deep nueral network model

Implementations:
Write comments
==================================================================================================='''

def main():
	X=np.random.rand(100,5);
	Y=np.ones((3,5));
	iter_nums=100;

	dim_layers=[100, 20, 3];
	activation_ls=['sigmoid','sigmoid'];
	nn_model(X, Y, iter_nums, dim_layers, activation_ls)
	# Exercise xyz
	# Write comments
	'''==============================================================================================
	Arguments:
	A -- activations from previous layer (or input data): (size of previous layer, number of examples)
	W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
	b -- bias vector, numpy array of shape (size of the current layer, 1)

	Returns:
	Z -- the input of the activation function, also called pre-activation parameter 
	cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
	=============================================================================================='''

def nn_model(X, Y, iter_nums, dim_layers, activation_ls, load_params=None, learning_rate=0.01):
	### Preparation Phrase
	cost=[];

	# Step 1: parameters
	if load_params==None:
		parameters=initialize_parameters(dim_layers, initializer='he')
	else :
		pass

	### Handling Phrase
	logging.getLogger().setLevel(logging.ERROR)
	for epochs in range(iter_nums):
		# Step 2: AL
		AL, caches=L_model_forward(X, parameters, activation_ls);
		# Step 3: grads
		grads=L_model_backward(AL, Y, caches, activation_ls);
		# Step 4: cost
		cost.append(compute_cost(AL, Y, parameters));
		if epochs%10==0:
			logging.getLogger().setLevel(logging.INFO);
			logging.info('Iter {} cost: {}'.format(epochs, cost[-1]));
			# logging.info('grad: {}'.format(grads['dW1']))
			logging.getLogger().setLevel(logging.ERROR);
		# Step 5: parameters
		parameters=update_parameters(parameters, grads, learning_rate=learning_rate);

	### Checking Phrase
	logging.info('Model training finished!')
	return parameters, cost


if __name__=='__main__':
	main()