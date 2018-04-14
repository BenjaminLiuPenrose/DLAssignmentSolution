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
from Implementations.sigmoid_activation import *

'''===================================================================================================
File content:
Implements the backward propagation for SIGMOID unit. You can call it as follows:
==================================================================================================='''

def sigmoid_backward(dA, activation_cache):
	'''==============================================================================================
	Arguments:
	dA -- Gradient of the cost with respect to the activation output (of the layer l)
	activation_cache -- dict of value (A_prev, W, b, Z) coming from the forward propagation in the current layer

	Returns:
	dZ -- Gradient of the cost with respect to the activation input (of the layer l), same shape as Z
	=============================================================================================='''
	# Preparation Phrase
	dZ=[];

	# Handling Phrase
	Z=activation_cache['Z'];
	A=sigmoid_activation(Z);
	g_prime=A*(1-A);

	dZ=dA*g_prime;

	# Checking Phrase
	assert(dZ.shape==Z.shape);
	return dZ