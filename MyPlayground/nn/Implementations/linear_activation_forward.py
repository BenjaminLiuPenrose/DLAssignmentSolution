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
logging.getLogger().setLevel(logging.DEBUG)
from Implementations.linear_forward import *
from Implementations.sigmoid_activation import *
from Implementations.relu_activation import *

'''===================================================================================================
File content:
Implement the forward propagation for the LINEAR->ACTIVATION layer, combine linear forward and activation
==================================================================================================='''

def linear_activation_forward(A_prev, W, b, activation='sigmoid'):
	'''==============================================================================================
	Arguments:
	A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
	W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
	b -- bias vector, numpy array of shape (size of the current layer, 1)
	activation -- the activation to be used in this layer, stored as a text string: "sigmoid" "tanh" or "relu"

	Returns:
	A -- the output of the activation function, also called the post-activation value 
	cache -- a python dictionary containing "linear_cache" and "activation_cache";
	         stored for computing the backward pass efficiently
	=============================================================================================='''
	# Preparation Phrase
	A=[];
	cache={};

	# Handling Phrase
	Z, cache=linear_forward(A_prev, W, b);

	if activation.lower()=='sigmoid':
		A=sigmoid_activation(Z);
	elif activation.lower()=='tanh':
		pass
	elif activation.lower()=='softmax':
		pass
	elif activation.lower()=='leaky relu':
		pass
	else :                                                              # activation.lower()=='relu'
		A=relu_activation(Z);

	# Checking Phrase
	return A, cache