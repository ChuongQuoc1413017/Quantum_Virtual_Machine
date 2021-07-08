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