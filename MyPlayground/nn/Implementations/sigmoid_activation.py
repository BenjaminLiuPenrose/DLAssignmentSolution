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
Implement the sigmoid activation function
==================================================================================================='''

def sigmoid_activation(Z):
	'''==============================================================================================
	Arguments:
	Z -- the input of the activation function, also called pre-activation parameter: (size of previous layer, number of examples)

	Returns:
	A -- activations for next layer previous layer
	==============================================================================================='''
	# Preparation Phrase
	A=[];

	# Handling Phrase
	A=1/(1+np.exp(-Z))

	# Checking Phrase
	return A 