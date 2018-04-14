# -*- coding: utf-8 -*- 
#!/usr/bin/env python
'''
Student name: Beier (Benjamin) Liu
Date:
Project Part

Remark:
Python 2.7 is recommended
Before running please install packages *numpy
Using cmd line py -2.7 -m install [package_name]
'''
import os, time, logging
import copy, math
import functools, itertools
import numpy as np 
logging.getLogger().setLevel(logging.INFO)

'''===================================================================================================
File content:
Implement the initialization of layers
==================================================================================================='''

def initialize_parameters(layer_dims, initializer='random'):
	'''===============================================================================================
	Arguments:
	layer_dims -- list containing the dimensions of each layer in our network
	initializer -- method of initialization includes random, he

	Returns:
	parameters -- dictionary containing parameters "W1", "b1", ..., "WL", "bL":
	                Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
	                bl -- bias vector of shape (layer_dims[l], 1)
	================================================================================================'''
	# Preparation Phrase
	parameters={};
	L=len(layer_dims)-1;

	# Handling Phrase
	for l in range(L):
		if initializer.lower()=='he':
			parameters['W{}'.format(l+1)]=np.random.rand(layer_dims[l+1], layer_dims[l])/np.sqrt(layer_dims[l]/2)
		elif initializer.lower()=='xavier':
			parameters['W{}'.format(l+1)]=np.random.rand(layer_dims[l+1], layer_dims[l])/np.sqrt(layer_dims[l])
		else :                                                          # initializer=='random'
			parameters['W{}'.format(l+1)]=np.random.rand(layer_dims[l+1], layer_dims[l]);
		parameters['b{}'.format(l+1)]=np.zeros((layer_dims[l+1],1));

	# Checking Phrase
	logging.info('Parameters initialization with {} finished!'.format(initializer.lower()))
	logging.debug('Parameters: {}'.format(parameters))
	return parameters