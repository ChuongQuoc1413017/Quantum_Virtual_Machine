# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:24:01 2021

@author: ASUS
"""
import numpy as np

class Wavefunction(object):
    """a wavefunction representing a quantum state"""

    def __init__(self, states, amplitude_vector):
        self.state = states
        self.amplitude = amplitude_vector
    
    def probabilities(self):
        """returns a dictionary of associated probabilities."""
        return np.abs(self.amplitude) ** 2

    def print_state(self):
        """represent a quantum state in bra-ket notations"""
        states = self.state
        string = str(self.amplitude[0]) + '|' + states[0] + '>'
        for i in range(1, len(states)):
            string += ' + ' + str(self.amplitude[i]) + '|' + states[i] + '>'
        return string

    def measure_sample(self, n_samples):
        """Make a measurement on quibits"""
        inds = np.random.choice(len(self.state), n_samples, p=self.probabilities())
        return np.unique(np.array(self.state[inds]), return_counts=True)