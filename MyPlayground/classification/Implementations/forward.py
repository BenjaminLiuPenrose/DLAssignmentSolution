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

def forward(X, parameters):
	'''==============================================================================================
	Arguments:
	X -- input data: (number of features, number of examples)
	parameters -- dictionary containing parameters "W", "b":
					W -- weight matrix of shape (K features, number of features)
					b -- bias vector of shape (K features, 1)

	Returns:
	EY -- estimated output: (K features, number of examples)
	cache -- dict containing activation value "Z"
					Z -- activation input value (K features, number of examples)
	=============================================================================================='''
	### Preparation Phrase
	### Handling Phrase
	### Checking Phrase