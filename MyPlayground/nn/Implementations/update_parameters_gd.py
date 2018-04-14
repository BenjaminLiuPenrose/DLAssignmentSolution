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

'''===================================================================================================
File content:
Update parameters using gradient descent
==================================================================================================='''

def update_parameters_gd(parameters, grads, learning_rate):
	'''==============================================================================================
	Arguments:
	parameters -- python dictionary containing your parameters 
	grads -- python dictionary containing your gradients, output of L_model_backward
	learning_rate -- the learning rate

	Returns:
	parameters -- python dictionary containing your updated parameters 
				  parameters["W" + str(l)] = ... 
				  parameters["b" + str(l)] = ...
	=============================================================================================='''
	# Preparation Phrase
	L=len(parameters)/2;

	# Handling Phrase
	for l in range(L):      
		parameters['W{}'.format(l+1)]= parameters['W{}'.format(l+1)]-learning_rate*grads['dW{}'.format(l+1)];
		parameters['b{}'.format(l+1)]= parameters['b{}'.format(l+1)]-learning_rate*grads['db{}'.format(l+1)];

	# Checking Phrase
	return parameters