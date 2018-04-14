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
logging.getLogger().setLevel(logging.DEBUG)
from Implementations.update_parameters_gd import *
from Implementations.update_parameters_adam import *

'''===================================================================================================
File content:
Update parameters using gradient descent
==================================================================================================='''

def update_parameters(parameters, grads, learning_rate, method='gd', h_params={}, mini_batch_size=64, velo={}, beta=1.0, beta1=0.9, beta2=0.999,  epsilon=1e-8):
	'''==============================================================================================
	Arguments:
	parameters -- python dictionary containing your parameters 
	grads -- python dictionary containing your gradients, output of L_model_backward
	learning_rate -- the learning rate
	method -- optimization method includes gd, minibatch, momentum, adam, rmsprop
	v -- Adam variable, moving average of the first gradient, python dictionary
	s -- Adam variable, moving average of the squared gradient, python dictionary
	beta1 -- Exponential decay hyperparameter for the first moment estimates 
	beta2 -- Exponential decay hyperparameter for the second moment estimates 
	epsilon -- hyperparameter preventing division by zero in Adam updates
	v -- python dictionary containing the current velocity:
					v['dW' + str(l)] = ...
					v['db' + str(l)] = ...
	beta -- the momentum hyperparameter, scalar

	Returns:
	parameters -- python dictionary containing your updated parameters 
				  parameters["W" + str(l)] = ... 
				  parameters["b" + str(l)] = ...
	=============================================================================================='''
	# Preparation Phrase
	'Pass'

	# Handling Phrase
	if method.lower()=='adam':
		parameters=update_parameters_adam(parameters, grads, learning_rate, h_params);
	elif method.lower()=='minibatch':
		pass
	elif method.lower()=='stochastic':
		pass
	elif method.lower()=='momentum':
		pass
	elif method.lower()=='rmsprop':
		pass
	else :                                                                      # method.lower=='dg'
		parameters=update_parameters_gd(parameters, grads, learning_rate);

	# Checking Phrase
	logging.info('Parameters updates with {} method finished!'.format(method));
	logging.debug('Parameters: {}'.format(parameters));
	return parameters