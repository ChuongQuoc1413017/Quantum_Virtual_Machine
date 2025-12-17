# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 17:08:42 2021

@author: ASUS
"""
import numpy as np
import math

def measure_all(wavefunction, n_samples):
    """Make a measurement on quibits"""
    inds = np.random.choice(len(wavefunction.state), n_samples, p=wavefunction.probabilities())
    return np.unique(np.array(wavefunction.state[inds]), return_counts=True)

def measure_one(wavefunction, n):
    """return a probability of |0> and |1> of qubit n"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    prob_0 = 0
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in range(2**qubit_num):
        if states[i][n] == '0':
            prob_0 += abs(amplitude[i])**2
    prob_0 = round(prob_0, 10)
    return np.array([prob_0, 1 - prob_0])

def collapse_one(wavefunction, n):
    """Measurement operator which make the Nth qubit collapse into |0> or |1>"""
    """regular error: https://stackoverflow.com/questions/48017053/numpy-random-choice-function-gives-weird-results"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    prob_0 = 0
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in range(2**qubit_num):
        if states[i][n] == '0':
            prob_0 += abs(amplitude[i])**2
    result_measure = np.random.choice(['0', '1'], 1, p = [prob_0, 1 - prob_0])[0]
    if result_measure == '0':
        for i in range(2**qubit_num):
            if states[i][n] == '0':
                new_amplitude[i] = amplitude[i]/math.sqrt(prob_0)
    elif result_measure == '1':
        for i in range(2**qubit_num):
            if states[i][n] == '1':
                new_amplitude[i] = amplitude[i]/math.sqrt(1 - prob_0)
    wavefunction.amplitude = new_amplitude

""" 
Updated on Tuesday Oct 14, 2025
"""
def pauli_expectation(wavefunction, qubit_idx, pauli_type):
    """
    Measure expectation value of Pauli operator on a specific qubit
    pauli_type: 'X', 'Y', or 'Z'
    """
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    
    if qubit_idx >= qubit_num or qubit_idx < 0:
        raise TypeError("Index is out of range")
    
    expectation = 0.0
    
    if pauli_type == 'X':
        # <X> = sum of amplitude[i]*conj(amplitude[j]) where state[i] and state[j] differ at qubit_idx
        cut = 2**(qubit_num - qubit_idx - 1)
        for i in range(2**qubit_num):
            if states[i][qubit_idx] == '0':
                # <i|X|j> where j is i with flipped bit at qubit_idx
                j = i + cut
                expectation += 2 * np.real(np.conj(amplitude[i]) * amplitude[j])
    
    elif pauli_type == 'Y':
        # <Y> similar to X but with imaginary component
        cut = 2**(qubit_num - qubit_idx - 1)
        for i in range(2**qubit_num):
            if states[i][qubit_idx] == '0':
                j = i + cut
                expectation += 2 * np.real(-1j * np.conj(amplitude[i]) * amplitude[j])
    
    elif pauli_type == 'Z':
        # <Z> = P(|0>) - P(|1>)
        for i in range(2**qubit_num):
            prob = abs(amplitude[i])**2
            if states[i][qubit_idx] == '0':
                expectation += prob
            else:
                expectation -= prob
    
    return expectation