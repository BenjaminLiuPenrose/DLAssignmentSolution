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
from Implementations.linear_activation_backward import *

'''===================================================================================================
File content:
Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
==================================================================================================='''

def L_model_backward(AL, Y, caches, parameters, activation_ls=None, regularization=False, lambd=0.0):
	'''==============================================================================================
	Arguments:
	AL -- probability vector, output of the forward propagation (L_model_forward())
	Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
	caches -- list of caches containing:
				every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
				the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
	parameters -- dict
	activation_ls -- list of activation method for each layer
	regularization -- boolean indicates regularize or not
	lambd -- if regularization, regularization coeffi lambda
	
	Returns:
	grads -- A dictionary with the gradients
			 grads["dA" + str(l)] = ... 
			 grads["dW" + str(l)] = ...
			 grads["db" + str(l)] = ... 
			 grads["dZ" + str(l)] = ... 
	=============================================================================================='''
	# Preparation Phrase
	grads={};
	L=len(caches);
	if activation_ls==None or len(activation_ls)!=L:
		activation_ls=['relu' for l in range(L)];
		activation_ls[-1]='sigmoid';

	# Handling Phrase
	cache=caches[L-1];
	dZ=AL-Y;
	dA_prev, dW, db=linear_backward(dZ, cache);

	grads['dA{}'.format(L)]=dA_prev; #This one is random 
	grads['dW{}'.format(L)]=dW;
	grads['db{}'.format(L)]=db;
	grads['dZ{}'.format(L)]=dZ;
	dA=dA_prev;
	# logging.info('AL: {}\n Y: {}\n dZ: {}'.format(AL, Y, dZ));

	for l in range(L-2,-1,-1):
		cache=caches[l];
		a=activation_ls[l];
		dA_prev, dW, db, dZ=linear_activation_backward(dA, cache, activation=a, regularization=regularization, lambd=lambd);
		# logging.info('dA_prev: {}\n dW: {}\n db: {}\n dZ: {}'.format(dA_prev, dW, db, dZ));

		grads['dA{}'.format(l+1)]=dA;
		grads['dW{}'.format(l+1)]=dW;
		grads['db{}'.format(l+1)]=db;
		grads['dZ{}'.format(l+1)]=dZ;
		dA=dA_prev;

	# Checking Phrase
	logging.info('Backward-pass with activation {} finished!'.format(activation_ls));
	logging.debug('grads: {}'.format(grads));
	return grads