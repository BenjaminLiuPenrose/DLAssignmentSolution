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
Implement the backward propagation gradient check
==================================================================================================='''
def gradient_check(x, theta, epsilon=1e-7):
	'''==============================================================================================    
    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    =============================================================================================='''
	print('\n====================================Exercise xyz=====================================\n');
	print('Running my myFunction function ... \n');
	myFunction();
	raw_input('Program pause. Press enter to continue.\n');