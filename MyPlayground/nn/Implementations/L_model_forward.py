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
from Implementations.linear_activation_forward import *

'''===================================================================================================
File content:
Write comments
==================================================================================================='''

def L_model_forward(X, parameters, activation_ls=None, dropout=False, keep_prob=1.0):
	'''==============================================================================================
	Arguments:
	X -- data, numpy array of shape (input size, number of examples)
	parameters -- output of initialize_parameters()
	dropout -- boolean value indicates whether or not apply dropout
	keep_prob -- if dropout, the probability of keep

	Returns:
	AL -- last post-activation value
	caches -- list of caches containing:
	            every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
	=============================================================================================='''
	# Preparation Phrase
	AL=[];
	caches=[];
	L=len(parameters)/2;
	if activation_ls==None or len(activation_ls)!=L:
		activation_ls=['relu' for l in range(L)];
		activation_ls[-1]='sigmoid';

	# Handling Phrase
	A_prev=X;
	for l in range(L):
		W=parameters['W{}'.format(l+1)];
		b=parameters['b{}'.format(l+1)];
		a=activation_ls[l];
		A, cache=linear_activation_forward(A_prev, W, b, activation=a);

		A_prev=A;
		caches.append(cache)
	AL=A;

	# Checking Phrase
	logging.info('Forward-pass with activation {} finished!'.format(activation_ls));
	logging.debug('AL : {}'.format(AL));
	logging.debug('caches: {}'.format(caches));
	return AL, caches
