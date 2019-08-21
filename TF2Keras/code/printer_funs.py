# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:29:50 2019

@author: KemyPeti
"""

import sys
import time

def percent_printer(iteration,
                    actual_step,
                    all_step,
                    string_ = ''):
    actual_step = actual_step + 1
    sys.stdout.write("\r" + string_ + "--Step:\t{0:7s}; ".format(str(iteration)) +
                     "[{}]".format((int(50*actual_step /all_step)) * '=' + ((50 - int(50*actual_step /all_step))* ' ')) + 
                     "{0:3s}%; ".format(str(int(100*actual_step /all_step))))
    sys.stdout.flush()

    