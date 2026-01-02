# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 18:30:28 2021

@author: ASUS
"""
from Qsun.Qwave import Wavefunction
from Qsun.Qcircuit import Qubit
from Qsun.Qgates import *
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

""" 
Updated on Thursday Jan 1, 2026
"""
def YZ_CX_encode(sample, params=None, n_layers=2):
    sample = np.array(sample)
    n_qubit = len(sample)
    expected_params = n_qubit * 2 * n_layers    
    if params is None:
        params = np.random.uniform(0, 2*np.pi, size=expected_params)
    else:
        params = np.array(params)
        if len(params) != expected_params:
            raise ValueError(
                f"Expected {expected_params} parameters "
                f"({n_qubit} qubits × 2 gates × {n_layers} layers), "
                f"got {len(params)}"
            )
    circuit_initial = Qubit(n_qubit)
    param_idx = 0
    for layer in range(n_layers):
        for q in range(n_qubit):
            RY(circuit_initial, q, phi=params[param_idx] + 2.0 * sample[q])
            param_idx += 1
            RZ(circuit_initial, q, phi=params[param_idx] + 2.0 * sample[q])
            param_idx += 1
        offset = layer % 2
        for i in range(offset, n_qubit - 1, 2):
            CNOT(circuit_initial, i, i + 1)
        if offset == 1 and n_qubit > 2:
            CNOT(circuit_initial, n_qubit - 1, 0)

    return circuit_initial

def HighDim_encode(sample):
    sample = np.array(sample)
    n_qubit = len(sample)
    circuit_initial = Qubit(n_qubit)
    for q in range(n_qubit):
        H(circuit_initial, q)
        i0 = (n_qubit - q + 0) % n_qubit  
        i1 = (n_qubit - q + 1) % n_qubit  
        i2 = (n_qubit - q + 2) % n_qubit  
        RZ(circuit_initial, q, phi=sample[i0])
        RY(circuit_initial, q, phi=sample[i1])
        RZ(circuit_initial, q, phi=sample[i2])
    for q in range(0, n_qubit - 1, 2):
        SISWAP(circuit_initial, q, q + 1)
    for q in range(1, n_qubit - 1, 2):
        SISWAP(circuit_initial, q, q + 1)
    for q in range(n_qubit):
        i0 = (n_qubit - q + 0) % n_qubit
        i1 = (n_qubit - q + 1) % n_qubit
        i2 = (n_qubit - q + 2) % n_qubit
        RZ(circuit_initial, q, phi=sample[i0])
        RY(circuit_initial, q, phi=sample[i1])
        RZ(circuit_initial, q, phi=sample[i2])

    return circuit_initial

def HZY_CZ_encode(sample, params=None, n_layers=2, closed=True):
    sample = np.array(sample)
    n_qubit = len(sample)
    feature_vector = np.array([sample[i % n_qubit] for i in range(n_qubit)])
    params_per_layer = n_qubit
    if n_qubit > 2:
        params_per_layer += n_qubit if closed else (n_qubit - 1)
    total_params = params_per_layer * n_layers
    if params is None:
        params = np.random.rand(total_params)
    else:
        params = np.asarray(params, dtype=float)
        if len(params) < total_params:
            raise ValueError(f"Need {total_params} params, got {len(params)}")
    circuit_initial = Qubit(n_qubit)
    param_idx = 0
    for q in range(n_qubit):
        H(circuit_initial, q)
    for layer in range(n_layers):
        for q in range(n_qubit):
            RZ(circuit_initial, q, feature_vector[q])
        for q in range(n_qubit):
            RY(circuit_initial, q, params[param_idx])
            param_idx += 1    
        if n_qubit > 2:
            istop = n_qubit if closed else (n_qubit - 1)
            for i in range(istop):
                CRZ(circuit_initial, i, (i + 1) % n_qubit, phi=params[param_idx])
                param_idx += 1
    return circuit_initial

def Chebyshev_encode(sample, params=None, n_layers=2):
    sample = np.array(sample)
    n_qubit = len(sample)
    circuit_initial = Qubit(n_qubit)
    crz_params_per_layer = n_qubit  
    total_num_params = (
        n_qubit + n_layers * (n_qubit + crz_params_per_layer) + n_qubit)
    if params is None:
        params = np.random.uniform(0, np.pi, size=total_num_params)
    else: 
        params = np.asarray(params, dtype=float)
    idx = 0
    for q in range(n_qubit):
        RY(circuit_initial, q, params[idx])
        idx += 1
    for layer in range(n_layers):
        for q in range(n_qubit):
            angle = params[idx] * np.arccos(np.clip(sample[q], -1, 1))
            RX(circuit_initial, q, angle)
            idx += 1
        for q in range(0, n_qubit - 1, 2):
            CRZ(circuit_initial, q, q + 1, phi=params[idx])
            idx += 1
        for q in range(1, n_qubit - 1, 2):
            CRZ(circuit_initial, q, q + 1, phi=params[idx])
            idx += 1
        CRZ(circuit_initial, n_qubit - 1, 0, phi=params[idx])
        idx += 1

    for q in range(n_qubit):
        RY(circuit_initial, q, params[idx])
        idx += 1
    
    return circuit_initial

def ParamZFeatureMap_encode(sample, params=None, n_layers=2):
    sample = np.array(sample)
    n_qubit = len(sample)
    circuit_initial = Qubit(n_qubit)
    num_params = n_layers * n_qubit
    if params is None:
        params = np.random.uniform(0, np.pi, size=num_params)
    else: 
        params = np.asarray(params, dtype=float)
    idx = 0
    for layer in range(n_layers):
        for q in range(n_qubit):
            H(circuit_initial, q)
            Phase(circuit_initial, q, phi=params[idx] * sample[q])
            idx += 1
        for q in range(0, n_qubit - 1, 1):
            CNOT(circuit_initial, q, q + 1)
    
    return circuit_initial

def SeparableRXEncoding_encode(sample):
    sample = np.array(sample)
    n_qubit = len(sample)
    circuit_initial = Qubit(n_qubit)
    for q in range(n_qubit):
        RX(circuit_initial, q, sample[q])
        RX(circuit_initial, q, sample[q])

    return circuit_initial

def HardwareEfficientEmbeddingRx_encode(sample, n_layers=2):
    sample = np.array(sample)
    n_qubit = len(sample)
    circuit_initial = Qubit(n_qubit)
    for layer in range(n_layers):
        for q in range(n_qubit):
            RX(circuit_initial, q, phi=sample[q])
        for q in range(0, n_qubit - 1, 1):
            CNOT(circuit_initial, q, q + 1)
    
    return circuit_initial

def ZFeatureMap_encode(sample, n_layers=2):
    sample = np.array(sample)
    n_qubit = len(sample)
    circuit_initial = Qubit(n_qubit)
    for layer in range(n_layers):
        for q in range(n_qubit):
            H(circuit_initial, q)
            Phase(circuit_initial, q, phi=2 * sample[q])
    
    return circuit_initial

def ZZFeatureMap_encode(sample, n_layers=2, entanglement="linear"):
    sample = np.array(sample)
    n_qubit = len(sample)
    circuit_initial = Qubit(n_qubit)
    for layer in range(n_layers):
        for i in range(n_qubit):
            H(circuit_initial, i)
        for i in range(n_qubit):
            Phase(circuit_initial, i, 2.0 * sample[i])
        if entanglement == "linear":
            for i in range(n_qubit - 1):
                angle = 2.0 * (np.pi - sample[i]) * (np.pi - sample[i+1])
                CNOT(circuit_initial, i, i+1)
                Phase(circuit_initial, i+1, angle)
                CNOT(circuit_initial, i, i+1)
        elif entanglement == "circular":
            for i in range(n_qubit - 1):
                angle = 2.0 * (np.pi - sample[i]) * (np.pi - sample[i+1])
                CNOT(circuit_initial, i, i+1)
                Phase(circuit_initial, i+1, angle)
                CNOT(circuit_initial, i, i+1)
            angle = 2.0 * (np.pi - sample[-1]) * (np.pi - sample[0])
            CNOT(circuit_initial, n_qubit-1, 0)
            Phase(circuit_initial, 0, angle)
            CNOT(circuit_initial, n_qubit-1, 0)
        elif entanglement == "full":
            for i in range(n_qubit):
                for j in range(i+1, n_qubit):
                    angle = 2.0 * (np.pi - sample[i]) * (np.pi - sample[j])
                    CNOT(circuit_initial, i, j)
                    Phase(circuit_initial, j, angle)
                    CNOT(circuit_initial, i, j)
    
    return circuit_initial
