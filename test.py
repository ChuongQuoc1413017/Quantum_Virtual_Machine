# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:31:34 2021

@author: ASUS
"""

from qCIT import qcit_circuit
from qCIT import qcit_gate

n = 2
circuit = qcit_circuit.Qubit(n)
qcit_gate.H(circuit, 0)
qcit_gate.CNOT(circuit, 0, 1)
print(circuit.print_state())
print()
print(circuit.measure_sample(10000))