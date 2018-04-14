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
from Implementations.linear_backward import *
from Implementations.sigmoid_backward import *
from Implementations.relu_backward import *

'''===================================================================================================
File content:
Implement the backward propagation for the LINEAR->ACTIVATION layer.
==================================================================================================='''

def linear_activation_backward(dA, cache, activation, regularization=False, lambd=0.0):
	'''==============================================================================================
	Arguments:
	dA -- post-activation gradient for current layer l 
	cache -- dict of values (A_prev, W, b, Z) we store for computing backward propagation efficiently
	activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
	regularization -- boolean indicates regularize or not
	lambd -- if regularization, regularization coeffi lambda
	
	Returns:
	dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	db -- Gradient of the cost with respect to b (current layer l), same shape as b
	dZ -- Gradient of the cost with respect to linear output/activation input (current layer l), same shape as b
	=============================================================================================='''
	# Preparation Phrase
	dA_prev=[]; 
	dW=[];
	db=[];
	dZ=[];

	# Handling Phrase
	if activation.lower()=='sigmoid':
		dZ=sigmoid_backward(dA, cache);
	elif activation.lower()=='tanh':
		pass
	elif activation.lower()=='softmax':
		pass
	elif activation.lower()=='leaky relu':
		pass
	else :                                                                         # activation.lower=='relu'
		dZ=relu_backward(dA, cache);
	
	dA_prev, dW, db=linear_backward(dZ, cache);

	# Checking Phrase
	A_prev, W, b, Z=cache['A_prev'], cache['W'], cache['b'], cache['Z'];
	assert(dA_prev.shape==A_prev.shape);
	assert(dW.shape==W.shape);
	assert(db.shape==b.shape);
	assert(dZ.shape==Z.shape);
	return dA_prev, dW, db, dZ
	
