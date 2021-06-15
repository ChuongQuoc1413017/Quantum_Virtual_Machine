# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:31:34 2021

@author: ASUS
"""

from qCIT import qCircuit
from qCIT import qBLAS

n = 2
circuit = qCircuit.Qubit(n)
qBLAS.H(circuit, 0)
qBLAS.CNOT(circuit, 0, 1)
print(circuit.print_state())
print()
print(circuit.measure_sample(10000))
print(circuit.visual_circuit())