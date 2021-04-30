# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:31:34 2021

@author: ASUS
"""

from QVM import qvm_circuit
from QVM import qvm_gate

n = 2
circuit = qvm_circuit.Qubit(n)
qvm_gate.H(circuit, 0)
qvm_gate.CNOT(circuit, 0, 1)
print(circuit.print_state())
print()
print(circuit.measure_sample(10000))