    # -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:26:23 2021

@author: ASUS
"""
import numpy as np
import math
import cmath

def shift_walk(wavefunction, dim):
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    new_amplitude = np.zeros(len(amplitude), dtype = complex)
    if dim != 1 and dim != 2:
        raise TypeError('The dimension of the quantum walk must be 1 or 2')
    if dim == 1:
        qubit_num = int((len(states) + 2)/4)
        for i in range(0, 2*qubit_num-2):
            new_amplitude[i+1] += amplitude[i]
        for i in range(2*(qubit_num), 4*(qubit_num)-2):
            new_amplitude[i-1] += amplitude[i]
    else:
        cut = int(len(wavefunction.state)/4)
        qubit_num = int(np.sqrt(len(wavefunction.state)/4))
        for i in range(0, qubit_num):
            for j in range(i*qubit_num, (i+1)*qubit_num-1):
                new_amplitude[j+1] += amplitude[j]
        for i in range(0, qubit_num-1):
            for j in range(i*qubit_num, (i+1)*qubit_num):
                new_amplitude[j+cut+qubit_num] += amplitude[j+cut]
        for i in range(1, qubit_num):
            for j in range(i*qubit_num, (i+1)*qubit_num):
                new_amplitude[j+2*cut-qubit_num] += amplitude[j+2*cut]
        for i in range(0, qubit_num):
            for j in range(i*qubit_num+1, (i+1)*qubit_num):
                new_amplitude[j+3*cut-1] += amplitude[j+3*cut]
    wavefunction.amplitude = new_amplitude
    
def H_coin(wavefunction, dim):
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    new_amplitude = np.zeros(len(amplitude), dtype = complex)
    if dim != 1 and dim != 2:
        raise TypeError('The dimension of the quantum walk must be 1 or 2')
    if dim == 1:
        qubit_num = int((len(states) + 2)/4)
        for i in np.nonzero(amplitude)[0]:
            if states[i][0] == '0':
                new_amplitude[i] += amplitude[i]/2**0.5
                new_amplitude[i+2*qubit_num-1] += amplitude[i]/2**0.5
            else:
                new_amplitude[i] -= amplitude[i]/2**0.5
                new_amplitude[i-2*qubit_num+1] += amplitude[i]/2**0.5
    else:
        cut = int(len(wavefunction.state)/4)
        for i in np.nonzero(amplitude)[0]:
            if states[i][0] == '0':
                new_amplitude[i] += amplitude[i]/2
                new_amplitude[i+cut] += amplitude[i]/2
                new_amplitude[i+2*cut] += amplitude[i]/2
                new_amplitude[i+3*cut] += amplitude[i]/2
            elif states[i][0] == '1':
                new_amplitude[i-cut] += amplitude[i]/2
                new_amplitude[i] -= amplitude[i]/2
                new_amplitude[i+cut] += amplitude[i]/2
                new_amplitude[i+2*cut] -= amplitude[i]/2
            elif states[i][0] == '2':
                new_amplitude[i-2*cut] += amplitude[i]/2
                new_amplitude[i-cut] += amplitude[i]/2
                new_amplitude[i] -= amplitude[i]/2
                new_amplitude[i+cut] -= amplitude[i]/2
            else:
                new_amplitude[i-3*cut] += amplitude[i]/2
                new_amplitude[i-2*cut] -= amplitude[i]/2
                new_amplitude[i-cut] -= amplitude[i]/2
                new_amplitude[i] += amplitude[i]/2
    wavefunction.amplitude = new_amplitude

def Gover_coin(wavefunction, dim):
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    new_amplitude = np.zeros(len(amplitude), dtype = complex)
    if dim != 2:
        raise TypeError('The dimension of the quantum walk must be 2')
    else:
        cut = int(len(wavefunction.state)/4)
        for i in np.nonzero(amplitude)[0]:
            if states[i][0] == '0':
                new_amplitude[i] -= amplitude[i]/2
                new_amplitude[i+cut] += amplitude[i]/2
                new_amplitude[i+2*cut] += amplitude[i]/2
                new_amplitude[i+3*cut] += amplitude[i]/2
            elif states[i][0] == '1':
                new_amplitude[i-cut] += amplitude[i]/2
                new_amplitude[i] -= amplitude[i]/2
                new_amplitude[i+cut] += amplitude[i]/2
                new_amplitude[i+2*cut] += amplitude[i]/2
            elif states[i][0] == '2':
                new_amplitude[i-2*cut] += amplitude[i]/2
                new_amplitude[i-cut] += amplitude[i]/2
                new_amplitude[i] -= amplitude[i]/2
                new_amplitude[i+cut] += amplitude[i]/2
            else:
                new_amplitude[i-3*cut] += amplitude[i]/2
                new_amplitude[i-2*cut] += amplitude[i]/2
                new_amplitude[i-cut] += amplitude[i]/2
                new_amplitude[i] -= amplitude[i]/2
    wavefunction.amplitude = new_amplitude
    
def quantum_walk_hadamard(wavefunction, dim, iteration):
    for i in range(iteration):
        H_coin(wavefunction, dim)
        shift_walk(wavefunction, dim)
        
def quantum_walk_grover(wavefunction, dim, iteration):
    for i in range(iteration):
        H_coin(wavefunction, dim)
        shift_walk(wavefunction, dim)