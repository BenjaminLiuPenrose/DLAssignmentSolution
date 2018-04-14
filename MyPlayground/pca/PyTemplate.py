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
Main program:
Write comments

Implementations:
Write comments

File content:
Write comments
==================================================================================================='''

def main():
	# Exercise xyz
	# Write comments
	'''==============================================================================================
	Arguments:
	A -- activations from previous layer (or input data): (size of previous layer, number of examples)
	W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
	b -- bias vector, numpy array of shape (size of the current layer, 1)

	Returns:
	Z -- the input of the activation function, also called pre-activation parameter 
	cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
	=============================================================================================='''
	### Preparation Phrase
	### Handling Phrase
	### Checking Phrase
	print('\n====================================Exercise xyz=====================================\n');
	print('Running my myFunction function ... \n');
	myFunction();
	raw_input('Program pause. Press enter to continue.\n');


if __name__=='__main__':
	main()


class Asset(object):
	# Class init
	# Object init
	def __ini__(self):
		pass

	# Getter and setter
	@property
	def get(self):
		pass

	@get.setter
	def set(self, i):
		pass

	# Static method
	@staticmethod
	def myFunc():
		pass

	# Class method
	@classmethod
	def myFunct(cls):
		pass

	# Object-level method
	def funct():
		pass