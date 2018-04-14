'''===================================================================================================
Benjamin's Python programming template file
You can ignore this file
==================================================================================================='''

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
Write comments
==================================================================================================='''

def compute_cost(EY, Y, parameters, regularization=False, lambd=0.0):
	'''==============================================================================================
	Arguments:
	EY -- estimated output: (K features, number of examples)
	Y -- output data: (K features, number of examples)
	parameters -- python dictionary containing parameters of the model
	regularization -- boolean indicates regularize or not
	lambd -- if regularization, regularization coeffi lambda

	Returns:
	Z -- the input of the activation function, also called pre-activation parameter 
	cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
	=============================================================================================='''
	### Preparation Phrase
	### Handling Phrase
	### Checking Phrase