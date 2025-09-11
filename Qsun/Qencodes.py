# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 18:30:28 2021

@author: ASUS
"""
from Qsun.Qwave import Wavefunction
from Qsun.Qcircuit import Qubit
from Qsun.Qgates import CNOT
import numpy as np
import math

def amplitude_encode(sample):
    '''Encode the who datapoint into the amplitude to qubits'''
    qubit_num = int(math.ceil(np.log2(len(sample))))
    circuit_initial = Qubit(qubit_num)
    circuit_initial.amplitude[0:len(sample)] = sample/np.sqrt(np.sum(sample**2))
    return circuit_initial

def qubit_encode(sample):
    '''Encode each feature into one qubit by using the rotation gate'''
    circuit_initial = Qubit(len(sample))
    ampli_vec = np.array([np.cos(sample[0]/2), -np.sin(sample[0]/2)])
    for i in range(1, len(sample)):
        ampli_vec = np.kron(ampli_vec, np.array([np.cos(sample[i]/2), -np.sin(sample[i]/2)]))
    circuit_initial.amplitude = ampli_vec
    return circuit_initial

def dense_encode(sample):
    '''Encode two features into one qubit by using the rotation gate'''
    qubit_num = int(len(sample)/2)
    circuit_initial = Qubit(qubit_num)
    ampli_vec = np.array([np.cos(sample[0+qubit_num]/2)*np.cos(sample[0]/2) - 1j*np.sin(sample[0+qubit_num]/2)*np.sin(sample[0]/2),
                          -np.sin(sample[0+qubit_num]/2)*np.cos(sample[0]/2) - 1j*np.cos(sample[0+qubit_num]/2)*np.sin(sample[0]/2)])
    for i in range(1, qubit_num):
        ampli_vec = np.kron(ampli_vec, np.array([np.cos(sample[i+qubit_num]/2)*np.cos(sample[i]/2) - 1j*np.sin(sample[i+qubit_num]/2)*np.sin(sample[i]/2),
                                      -np.sin(sample[i+qubit_num]/2)*np.cos(sample[i]/2) - 1j*np.cos(sample[i+qubit_num]/2)*np.sin(sample[i]/2)]))
    circuit_initial.amplitude = ampli_vec
    return circuit_initial

def unit_encode(sample):
    '''Encode each feature into one qubit's amplitude by using the square root function'''
    circuit_initial = Qubit(len(sample))
    ampli_vec = np.array([np.sqrt(sample[0]), np.sqrt(1-sample[0])])
    for i in range(1, len(sample)):
        ampli_vec = np.kron(ampli_vec, np.array([np.sqrt(sample[i]), np.sqrt(1-sample[i])]))
    circuit_initial.amplitude = ampli_vec
    return circuit_initial

def entangle(circuit, entanglement):
    ''' Add entanglement to the qubits'''
    circuit_layer = circuit
    qubit_num = int(math.ceil(np.log2(len(circuit_layer.state))))
    if entanglement == "linear":
        for i in range(qubit_num - 1):
            CNOT(circuit_layer, i, i + 1)
    elif entanglement == "circular":
        for i in range(qubit_num - 1):
            CNOT(circuit_layer, i, i + 1)
        CNOT(circuit_layer, qubit_num - 1, 0)
    elif entanglement == "full":
        for i in range(qubit_num):
            for j in range(i + 1, qubit_num):
                CNOT(circuit_layer, i, j)
    return circuit_layer