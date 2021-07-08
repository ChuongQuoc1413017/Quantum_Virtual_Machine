# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:31:34 2021

@author: ASUS
"""

from Qsun import Qcircuit
from Qsun import Qgates
from Qsun import Qmeas

n = 2
circuit = Qcircuit.Qubit(n)
Qgates.H(circuit, 0)
Qgates.CNOT(circuit, 0, 1)

print('state:')
print(circuit.print_state())
print()

print('measure circuit:')
print(Qmeas.measure_all(circuit, 10000))
print()

print('probabilities of qubit 1:')
print(Qmeas.measure_one(circuit, 1))
print()

print('visualization:')
print(circuit.visual_circuit())