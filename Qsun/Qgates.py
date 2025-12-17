    # -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:26:23 2021

@author: ASUS
"""
import numpy as np
import cmath

def H(wavefunction, n):
    """Hadamard gate"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-n-1)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in np.nonzero(amplitude)[0]:
        if states[i][n] == '0':
            new_amplitude[i] += amplitude[i]/2**0.5
            new_amplitude[i+cut] += amplitude[i]/2**0.5
        else:
            new_amplitude[i] -= amplitude[i]/2**0.5
            new_amplitude[i-cut] += amplitude[i]/2**0.5  
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([n, 'H'])
    
def X(wavefunction, n):
    """Pauli-X"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-n-1)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in np.nonzero(amplitude)[0]:
        if states[i][n] == '0':
            new_amplitude[i+cut] += amplitude[i]
        else:
            new_amplitude[i-cut] += amplitude[i]  
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([n, 'X'])
    
def Y(wavefunction, n):
    """Pauli-Y"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-n-1)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in np.nonzero(amplitude)[0]:
        if states[i][n] == '0':
            new_amplitude[i+cut] += 1.0j*amplitude[i]
        else:
            new_amplitude[i-cut] -= 1.0j*amplitude[i]  
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([n, 'Y'])
    
def Z(wavefunction, n):
    """Pauli-Z"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in np.nonzero(amplitude)[0]:
        if states[i][n] == '0':
            new_amplitude[i] += amplitude[i]
        else:
            new_amplitude[i] -= amplitude[i]  
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([n, 'Z'])
    
def RX(wavefunction, n, phi=0):
    """Rotation around X-axis gate"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-n-1)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in np.nonzero(amplitude)[0]:
        if states[i][n] == '0':
            new_amplitude[i] += cmath.cos(phi/2)*amplitude[i]
            new_amplitude[i+cut] -= 1j*cmath.sin(phi/2)*amplitude[i]
        else:
            new_amplitude[i] += cmath.cos(phi/2)*amplitude[i]
            new_amplitude[i-cut] -= 1j*cmath.sin(phi/2)*amplitude[i] 
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([n, 'RX', '0'])
    
def RY(wavefunction, n, phi=0):
    """Rotation around Y-axis gate"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-n-1)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in np.nonzero(amplitude)[0]:
        if states[i][n] == '0':
            new_amplitude[i] += cmath.cos(phi/2)*amplitude[i]
            new_amplitude[i+cut] += cmath.sin(phi/2)*amplitude[i]
        else:
            new_amplitude[i] += cmath.cos(phi/2)*amplitude[i]
            new_amplitude[i-cut] -= cmath.sin(phi/2)*amplitude[i] 
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([n, 'RY', '0'])
    
def RZ(wavefunction, n, phi=0):
    """Rotation around Z-axis gate"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in np.nonzero(amplitude)[0]:
        if states[i][n] == '0':
            new_amplitude[i] += cmath.exp(-1j*phi/2)*amplitude[i]
        else:
            new_amplitude[i] += cmath.exp(1j*phi/2)*amplitude[i]  
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([n, 'RZ', '0'])
""" 

Updated on Tuesday Oct 14, 2025

"""
def CRX(wavefunction, control, target, phi=0):
    """ Controlled RX gate - implements controlled rotation around X-axis."""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype=complex)
    cut = 2**(qubit_num-target-1)
    if control >= qubit_num or control < 0 or target >= qubit_num or target < 0:
        raise TypeError("Index is out of range.")
    if control == target:
        raise TypeError("Control qubit and target qubit must be distinct.")
    for i in np.nonzero(amplitude)[0]:
        if states[i][control] == '1':
            if states[i][target] == '0':
                new_amplitude[i] += cmath.cos(phi/2)*amplitude[i]
                new_amplitude[i+cut] -= 1j*cmath.sin(phi/2)*amplitude[i]
            else:
                new_amplitude[i] += cmath.cos(phi/2)*amplitude[i]
                new_amplitude[i-cut] -= 1j*cmath.sin(phi/2)*amplitude[i]
        else:
            new_amplitude[i] = amplitude[i]
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([control, target, 'CRX', '0'])

def CRY(wavefunction, control, target, phi=0):
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype=complex)
    cut = 2**(qubit_num-target-1)
    if control >= qubit_num or control < 0 or target >= qubit_num or target < 0:
        raise TypeError("Index is out of range.")
    if control == target:
        raise TypeError("Control qubit and target qubit must be distinct.")
    for i in np.nonzero(amplitude)[0]:
        if states[i][control] == '1':
            if states[i][target] == '0':
                new_amplitude[i] += cmath.cos(phi/2)*amplitude[i]
                new_amplitude[i+cut] += cmath.sin(phi/2)*amplitude[i]
            else:
                new_amplitude[i] += cmath.cos(phi/2)*amplitude[i]
                new_amplitude[i-cut] -= cmath.sin(phi/2)*amplitude[i]
        else:
            new_amplitude[i] = amplitude[i]

    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([control, target, 'CRY', '0'])

def CRZ(wavefunction, control, target, phi=0):
    """Controlled RZ gate - implements controlled rotation around Z-axis"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype=complex)
    if control >= qubit_num or control < 0 or target >= qubit_num or target < 0:
        raise TypeError("Index is out of range")
    if control == target:
        raise TypeError("Control qubit and target qubit must be distinct")
    for i in np.nonzero(amplitude)[0]:
        if states[i][control] == '1':
            # Apply RZ rotation on target qubit
            if states[i][target] == '0':
                new_amplitude[i] += cmath.exp(-1j*phi/2) * amplitude[i]
            else:
                new_amplitude[i] += cmath.exp(1j*phi/2) * amplitude[i]
        else:
            new_amplitude[i] = amplitude[i]
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([control, target, 'CRZ', '0'])

def Phase(wavefunction, n, phi=0):
    """PHASE gate"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in np.nonzero(amplitude)[0]:
        if states[i][n] == '0':
            new_amplitude[i] += amplitude[i]
        else:
            new_amplitude[i] += cmath.exp(1j*phi)*amplitude[i]  
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([n, 'P'])
#     (wavefunction.visual).append([n, 'P', phi])
    
def S(wavefunction, n):
    """Phase(pi/2)"""
    Phase(wavefunction, n , cmath.pi/2)
    (wavefunction.visual).append([n, 'S'])
    
def T(wavefunction, n):
    """Phase(pi/4)"""
    Phase(wavefunction, n , cmath.pi/4)
    (wavefunction.visual).append([n, 'T'])
    
def Xsquare(wavefunction, n):
    """a square root of the NOT gate."""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-n-1)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in np.nonzero(amplitude)[0]:
        new_amplitude[i] += (1+1j)*amplitude[i]/2
        if states[i][n] == '0':
            new_amplitude[i+cut] += (1-1j)*amplitude[i]/2
        else:
            new_amplitude[i-cut] += (1-1j)*amplitude[i]/2  
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([n, 'XS'])
    
def CNOT(wavefunction, control, target):
    """Flip target if control is |1>"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    if control < target or control > target:
        cut = 2**(qubit_num-target-1)
    else:
        raise TypeError("Control qubit and target qubit must be distinct")
    for i in np.nonzero(amplitude)[0]:
        if states[i][control] == '1':
            if states[i][target] == '0':
                new_amplitude[i+cut] += amplitude[i]
            else:
                new_amplitude[i-cut] += amplitude[i]
        else:
            new_amplitude[i] = amplitude[i]
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([control, target, 'CX'])
    
def CPhase(wavefunction, control, target, phi=0):
    """Controlled PHASE gate"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    if control == target:
        raise TypeError("Control qubit and target qubit must be distinct")
    for i in np.nonzero(amplitude)[0]:
        if states[i][control] == '1':
            if states[i][target] == '0':
                new_amplitude[i] += amplitude[i]
            else:
                new_amplitude[i] += cmath.exp(1j*phi)*amplitude[i] 
        else:
            new_amplitude[i] = amplitude[i]
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([control, target, 'CP', '0'])
    
def CCNOT(wavefunction, control_1, control_2, target):
    """CCNOT - double-controlled-X"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-target-1)
    if control_1 == target or control_2 == target or control_1 == control_2:
        raise TypeError("Control qubit and target qubit must be distinct")
    for i in np.nonzero(amplitude)[0]:
        if states[i][control_1] == '1' and states[i][control_2] == '1':
            if states[i][target] == '0':
                new_amplitude[i+cut] += amplitude[i]
            else:
                new_amplitude[i-cut] += amplitude[i]
        else:
            new_amplitude[i] = amplitude[i]
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([control_1, control_2, target, 'CCX'])
    
def OR(wavefunction, control_1, control_2, target):
    """CCNOT - double-controlled-X"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-target-1)
    if control_1 == target or control_2 == target or control_1 == control_2:
        raise TypeError("Control qubit and target qubit must be distinct")
    for i in np.nonzero(amplitude)[0]:
        if states[i][control_1] == '1' or states[i][control_2] == '1':
            if states[i][target] == '0':
                new_amplitude[i+cut] += amplitude[i]
            else:
                new_amplitude[i-cut] += amplitude[i]
        else:
            new_amplitude[i] = amplitude[i]
    wavefunction.amplitude = new_amplitude
    
def SWAP(wavefunction, target_1, target_2):
    """Swap gate"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    minimum = target_2 ^ ((target_1 ^ target_2) & -(target_1 < target_2))
    maximum = target_1 ^ ((target_1 ^ target_2) & -(target_1 < target_2)) 
    cut = 2**(qubit_num-minimum-1) - 2**(qubit_num-maximum-1)
    if target_1 == target_2:
        raise TypeError("Target qubits must be distinct")
    for i in range(2**qubit_num):
        if states[i][target_1] != states[i][target_2]:
            if int(states[i][maximum]) > int(states[i][minimum]):
#                 print(states[i], 'to', states[i+cut])
                new_amplitude[i+cut] += amplitude[i]                              
            else:
#                 print(states[i], 'to', states[i-cut])
                new_amplitude[i-cut] += amplitude[i]
        else:
#                 print(states[i], 'to', states[i])
            new_amplitude[i] = amplitude[i]
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([target_1, target_2, 'SWAP'])

def CSWAP(wavefunction, control, target_1, target_2):
    """CSwap gate"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    minimum = target_2 ^ ((target_1 ^ target_2) & -(target_1 < target_2))
    maximum = target_1 ^ ((target_1 ^ target_2) & -(target_1 < target_2)) 
    cut = 2**(qubit_num-minimum-1) - 2**(qubit_num-maximum-1)
    if control == target_1 or control == target_2 or target_1 == target_2:
        raise TypeError("Control qubit and target qubit must be distinct")
    for i in range(2**qubit_num):
        if states[i][control] == '1':
            if states[i][target_1] != states[i][target_2]:
                if int(states[i][maximum]) > int(states[i][minimum]):
    #                 print(states[i], 'to', states[i+cut])
                    new_amplitude[i+cut] += amplitude[i]                              
                else:
    #                 print(states[i], 'to', states[i-cut])
                    new_amplitude[i-cut] += amplitude[i]
            else:
    #                 print(states[i], 'to', states[i])
                new_amplitude[i] = amplitude[i]
        else:
            new_amplitude[i] = amplitude[i]
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([target_1, target_2, control, 'CSWAP'])
""" 

Updated on Sunday Dec 07, 2025

"""

def ISWAP(wavefunction, target_1, target_2):
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype=complex)
    if target_1 == target_2:
        raise TypeError("Target qubits must be distinct")
    minimum = target_2 ^ ((target_1 ^ target_2) & -(target_1 < target_2))
    maximum = target_1 ^ ((target_1 ^ target_2) & -(target_1 < target_2))
    cut = 2**(qubit_num-minimum-1) - 2**(qubit_num-maximum-1)
    dim = 2**qubit_num
    for i in range(dim):
        if states[i][target_1] == states[i][target_2]:
            new_amplitude[i] += amplitude[i]
        else:
            if int(states[i][maximum]) > int(states[i][minimum]):
                j = i + cut
            else:
                j = i - cut
            new_amplitude[j] += 1j * amplitude[i]
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([target_1, target_2, 'ISWAP'])

def SISWAP(wavefunction, target_1, target_2):
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype=complex)
    if target_1 == target_2:
        raise TypeError("Target qubits must be distinct")
    minimum = target_2 ^ ((target_1 ^ target_2) & -(target_1 < target_2))
    maximum = target_1 ^ ((target_1 ^ target_2) & -(target_1 < target_2))
    cut = 2**(qubit_num-minimum-1) - 2**(qubit_num-maximum-1)
    dim = 2**qubit_num
    processed = np.zeros(dim, dtype=bool)  
    c = 1.0 / np.sqrt(2.0)
    for i in range(dim):
        if processed[i]:
            continue
        b1 = states[i][target_1]
        b2 = states[i][target_2]
        if b1 == b2:
            new_amplitude[i] = amplitude[i]
            processed[i] = True
        else:
            if int(states[i][maximum]) > int(states[i][minimum]):
                j = i + cut
            else:
                j = i - cut
            a = amplitude[i]
            b = amplitude[j]
            new_amplitude[i] = c * a + 1j * c * b
            new_amplitude[j] = 1j * c * a + c * b
            processed[i] = True
            processed[j] = True
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([target_1, target_2, 'SISWAP'])

def E(wavefunction, p, n):
    """Quantum depolarizing channel"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-n-1)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in np.nonzero(amplitude)[0]:
        if states[i][n] == '0':
            new_amplitude[i+cut] += (p/2)*abs(amplitude[i])**2
            new_amplitude[i] += (1-p/2)*abs(amplitude[i])**2
        else:
            new_amplitude[i-cut] += (p/2)*abs(amplitude[i])**2
            new_amplitude[i] += (1-p/2)*abs(amplitude[i])**2
    #     wavefunction.wave.iloc[0, :] = np.sqrt(new_amplitude)
    for i in range(2**qubit_num):
        if amplitude[i] < 0:
            new_amplitude[i] = - np.sqrt(new_amplitude[i])
        else:
            new_amplitude[i] = np.sqrt(new_amplitude[i])
    wavefunction.amplitude = new_amplitude

def E_all(wavefunction, p_noise, qubit_num):
    if p_noise > 0:
        for i in range(qubit_num):
            E(wavefunction, p_noise, i)