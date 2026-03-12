# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 17:10:08 2021

@author: ASUS
"""
from Qsun.Qwave import Wavefunction
from Qsun.Qmeas import measure_one
from Qsun.Qgates import H, CSWAP, CNOT
from Qsun.Qmeas import pauli_expectation
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


""" 
Updated on Tuesday Oct 14, 2025
"""
class ProjectedQuantumKernel:
    def __init__(self, 
                circuit_fn,
                n_qubits=4, 
                n_layers=3,
                params= None, 
                gamma=1.0, 
                random_state=42):
        """
        Parameters:
        -----------
        q_circuit_fn : function, optional
            Custom quantum circuit function. 
        n_qubits : int
            Number of qubits
        n_layers : int
            Number of layers in the circuit
        gamma : float, default=1.0
            Gamma multiplier for RBF kernel
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.circuit = circuit_fn
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params = np.array(params) if params is not None else None
        self.gamma = gamma
        self.random_state = random_state
    
    def set_params(self, params):
        """
        Parameters:
        -----------
        params : array-like
            Circuit parameters
        
        Returns:
        --------
        self : ProjectedQuantumKernel
        """
        self.params = np.array(params)
        return self
    
    def compute_quantum_features(self, X):
        """
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        
        Returns:
        --------
        features : ndarray, shape (n_samples, n_qubits * 3)
            Quantum features (expectation values)
        """
        """if self.params is None:
            raise ValueError("Parameters are not initialized!.")"""
        
        exp_vals_list = []
        
        for x in X:
            exp_vals = self.circuit(x, self.params)
            exp_vals_list.append(np.array(exp_vals))
        
        features = np.array(exp_vals_list)
        return features
    
    def kernel_matrix(self, X1, X2=None):
        """
        Parameters:
        -----------
        X1 : array-like, shape (n_samples_1, n_features)
            First dataset
        X2 : array-like, shape (n_samples_2, n_features), optional
            Second dataset. If None, uses X1.
        
        Returns:
        --------
        K : ndarray, shape (n_samples_1, n_samples_2)
            Kernel matrix
        """
        if X2 is None:
            X2 = X1
        
        features_1 = self.compute_quantum_features(X1)
        features_2 = self.compute_quantum_features(X2)
        
        features_1 = features_1.reshape(len(X1), 3, self.n_qubits)
        features_2 = features_2.reshape(len(X2), 3, self.n_qubits)
        
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                diff_squared = np.sum((features_1[i] - features_2[j])**2)
                K[i, j] = np.exp(-self.gamma * diff_squared)
        
        return K
    
    def __call__(self, X1, X2):
        """
        Parameters:
        -----------
        X1 : array-like
            First dataset
        X2 : array-like
            Second dataset
        
        Returns:
        --------
        K : ndarray
            Kernel matrix
        """
        return self.kernel_matrix(X1, X2)

""" 
Updated on Saturday Dec 6, 2025
"""
def hadamard_test(wavefunction_1, wavefunction_2):
    amplitude_1 = wavefunction_1.amplitude
    amplitude_2 = wavefunction_2.amplitude
    state_1 = wavefunction_1.state
    state_2 = wavefunction_2.state
    qubit_num_1 = len(state_1[0])
    qubit_num_2 = len(state_2[0])
    if qubit_num_1 != qubit_num_2:
        raise TypeError('The number of qubits in both states must be equal')
    n = qubit_num_1
    total_qubits = 1 + 2*n  
    states = ["".join(seq) for seq in itertools.product("01", repeat=total_qubits)]
    amplitude_vector = np.zeros(2**total_qubits, dtype=complex)
    for i in range(len(state_1)):
        for j in range(len(state_2)):
            composite_state = '0' + state_1[i] + state_2[j]
            idx = int(composite_state, 2)
            amplitude_vector[idx] = amplitude_1[i] * amplitude_2[j]
    test = Wavefunction(np.array(states), amplitude_vector)
    H(test, 0)  
    for i in range(n):
        control_qubit = 0
        target1 = i + 1
        target2 = i + n + 1
        CNOT(test, control_qubit, target1)
        CNOT(test, target1, target2)
        CNOT(test, control_qubit, target1)
    H(test, 0)   
    return pauli_expectation(test, 0, 'Z')
