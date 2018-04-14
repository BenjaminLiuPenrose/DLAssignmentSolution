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

'''===================================================================================================
File content:
Implement the linear portion of backward propagation for a single layer (layer l)
==================================================================================================='''
def linear_backward(dZ, cache):
	'''==============================================================================================
	Arguments:
	dZ -- Gradient of the cost with respect to the linear output (of current layer l)
	cache -- dict of values (A_prev, W, b, Z) coming from the forward propagation in the current layer

	Returns:
	dA_prev -- Gradient of the cost with respect to the activation/linear input (of the previous layer l-1), same shape as A_prev
	dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	db -- Gradient of the cost with respect to b (current layer l), same shape as b
	=============================================================================================='''
	# Preparation Phrase
	dA_prev=[];
	dW=[];
	db=[];
	m=dZ.shape[1]

	# Handling Phrase
	A_prev, W, b=cache['A_prev'], cache['W'], cache['b'];
	dA_prev=np.matmul(W.T, dZ);
	dW=1.0/m*np.matmul(dZ, A_prev.T);     # logging.info('dZ: {}\n A_prev: {}\n dW: {}'.format(dZ, dA_prev, np.matmul(dZ, A_prev.T)));
	db=1.0/m*np.sum(dZ, axis=1, keepdims=True);
	
	# Checking Phrase
	assert(dA_prev.shape==A_prev.shape);
	assert(dW.shape==W.shape);
	assert(db.shape==b.shape);
	return dA_prev, dW, db 
