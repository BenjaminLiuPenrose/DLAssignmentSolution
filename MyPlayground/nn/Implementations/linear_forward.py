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
Implement the linear part of a layer's forward propagation.
==================================================================================================='''

def linear_forward(A, W, b):
	'''==============================================================================================
	Arguments:
	A -- activations from previous layer (or input data): (size of previous layer, number of examples)
	W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
	b -- bias vector, numpy array of shape (size of the current layer, 1)

	Returns:
	Z -- the input of the activation function, also called pre-activation parameter 
	cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
	=============================================================================================='''
	# Preparation Phrase
	Z=[];
	cache={}

	# Handling Phrase
	Z=np.matmul(W, A)+b;

	cache['A_prev']=A;
	cache['W']=W;
	cache['b']=b;
	cache['Z']=Z;

	# Checking Phrase
	return Z, cache
