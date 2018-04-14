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
Implements the backward propagation for SIGMOID unit. You can call it as follows:
==================================================================================================='''

def sigmoid_backward(dA, activation_cache):
	'''==============================================================================================
    Arguments:
    dA -- 
    activation_cache -- 

    Returns:
    dZ -- 
    =============================================================================================='''
	print('\n====================================Exercise xyz=====================================\n');
	print('Running my myFunction function ... \n');
	myFunction();
	raw_input('Program pause. Press enter to continue.\n');