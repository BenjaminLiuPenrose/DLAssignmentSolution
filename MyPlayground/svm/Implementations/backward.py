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

def backward(EY, Y, cache, parameters, regularization=False, lambd=0.0):
	'''==============================================================================================
	Arguments:
	EY -- estimated output: (K features, number of examples)
	Y -- output data: (K features, number of examples)
	cache -- dict containing activation value "Z"
					Z -- activation input value (K features, number of examples)
	parameters -- python dictionary containing parameters of the model
	regularization -- boolean indicates regularize or not
	lambd -- if regularization, regularization coeffi lambda
	
	Returns:
	grads -- dictionary containing grads "dW", "db":
					dW -- weight matrix of shape (K features, number of features)
					db -- bias vector of shape (K features, 1)
	=============================================================================================='''
	### Preparation Phrase
	### Handling Phrase
	### Checking Phrase