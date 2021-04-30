# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:25:18 2021

@author: ASUS
"""
from QVM.qvm_wave import Wavefunction
import itertools
import numpy as np

# https://stackoverflow.com/questions/4928297/all-permutations-of-a-binary-sequence-x-bits-long
def Qubit(qubit_num):
    """create a quantum circuit"""
    states = ["".join(seq) for seq in itertools.product("01", repeat=qubit_num)]
    amplitude_vector = np.zeros(2**qubit_num, dtype = complex)
    amplitude_vector[0] = 1.0
    return Wavefunction(np.array(states), amplitude_vector)