# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:31:34 2021

@author: ASUS
"""

from qSUN import qCircuit
from qSUN import qGATES
from qSUN import qMEAS

n = 2
circuit = qCircuit.Qubit(n)
qGATES.H(circuit, 0)
qGATES.CNOT(circuit, 0, 1)

print('state:')
print(circuit.print_state())
print()

print('measure circuit:')
print(qMEAS.measure_all(circuit, 10000))
print()

print('probabilities of qubit 1:')
print(qMEAS.prob_qubit(circuit, 1))
print()

print('visualization:')
print(circuit.visual_circuit())