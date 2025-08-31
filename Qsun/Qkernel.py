# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 17:10:08 2021

@author: ASUS
"""
from Qsun.Qwave import Wavefunction
from Qsun.Qmeas import measure_one
from Qsun.Qgates import H, CSWAP
import itertools
import numpy as np
import math

def state_product(wavefunction_1, wavefunction_2):
    """Return a quantum kernel of two quantum states"""
    amplitude_1 = wavefunction_1.amplitude
    amplitude_2 = wavefunction_2.amplitude
    return (np.absolute(np.vdot(amplitude_1, amplitude_2)))

def swap_test(wavefunction_1, wavefunction_2):
    """Return a quantum kernel of two quantum states using the Swap Test"""
    amplitude_1 = wavefunction_1.amplitude
    amplitude_2 = wavefunction_2.amplitude
    state_1 = wavefunction_1.state
    state_2 = wavefunction_2.state
    qubit_num_1 = len(state_1[0])
    qubit_num_2 = len(state_2[0])
    if qubit_num_1 != qubit_num_2:
        raise TypeError('The number of qubits in both states must be equal')
    states = ["".join(seq) for seq in itertools.product("01", repeat=qubit_num_1+qubit_num_2+1)]
    amplitude_vector = np.zeros(2**(qubit_num_1 + qubit_num_2+1), dtype = complex)
    for i in range(len(state_1)):
        for j in range(len(state_2)):
            amplitude_vector[int(state_1[i] + state_2[j], 2)] = amplitude_1[i]*amplitude_2[j]
    test =  Wavefunction(np.array(states), amplitude_vector)
    H(test, 0)
    for i in range(qubit_num_1):
        CSWAP(test, 0, i+1, i+qubit_num_1+1)
    H(test, 0)
    result = measure_one(test, 0)
    return np.sqrt(2*result[0]-1)