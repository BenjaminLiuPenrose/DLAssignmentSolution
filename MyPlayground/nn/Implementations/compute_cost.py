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
Implement the cost function defined by cross-entropy
==================================================================================================='''

def compute_cost(AL, Y, parameters, regularization=False, lambd=0.0):
	'''==============================================================================================
	Arguments:
	AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
	Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
	parameters -- python dictionary containing parameters of the model
	regularization -- boolean indicates regularize or not
	lambd -- if regularization, regularization coeffi lambda

	Returns:
	cost -- cross-entropy cost (with or without regularization)
	=============================================================================================='''
	# Preparation Phrase
	cost=0;
	m=Y.shape[1];

	# Handling Phrase
	cost=np.sum(Y*np.log(AL));
	cost=-1.0/m*cost;

	# Checking Phrase
	return cost
