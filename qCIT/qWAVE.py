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
        self.visual = []
    
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
    
    def visual_circuit(self):
        """Visualization of a ciruict"""
        n = len((self.state)[0])
        a = self.visual
        b = [[]]*(2*n)
        for i in range(2*n):
            b[i] = [0]*len(a)

        for i in range(n):
            for j in range(len(a)):
                if i in a[j]:    
                    if ('RX' in a[j]) or ('RY' in a[j]) or ('RZ' in a[j]):
                        b[2*i][j] = 1.5
                    elif ('CRX' in a[j]) or ('CRY' in a[j]) or ('CRZ' in a[j]):
                        b[2*i][j] = 2.5
                    elif ('CX' in a[j]) or ('SWAP' in a[j]):
                        b[2*i][j] = 3
                    elif ('CP' in a[j]):
                        b[2*i][j] = 3.5
                    elif ('CCX' in a[j]):
                        b[2*i][j] = 4
                    else:
                        b[2*i][j] = 1

        for j in range(len(a)):
            if ('CX' in a[j]) or ('CCX' in a[j]) or ('SWAP' in a[j]):
                for i in range(2*min(a[j][:-1])+1, 2*max(a[j][:-1]), 2):
                    b[i][j] = 2
            if ('CP' in a[j]) or ('CRX' in a[j]):
                for i in range(2*min(a[j][:-2])+1, 2*max(a[j][:-2]), 2):
                    b[i][j] = 2

        string_out = [[]]*(2*n)
        for i in range(2*n):
            string_out[i] = []

        for i in range(n):
            out = ''
            if i < 10:
                out += '|Q_'+str(i)+'> : '
            else:
                out += '|Q_'+str(i)+'>: '
            space = ' '*len(out)
            string_out[2*i].append(out)
            string_out[2*i+1].append(space)

            out = ''
            space = ''
            for j in range(len(a)):

                if b[2*i][j] == 0:
                    out += '---'

                if b[2*i][j] == 1:
                    out += a[j][-1] + '--'

                if b[2*i][j] == 1.5:
                    out += a[j][-2] + '-'

                if b[2*i][j] == 2.5:
                    if i == a[j][0]:
                        out += 'o--'
                    elif i == a[j][1]:
                        out += a[j][-2][1:] + '-'

                if b[2*i][j] == 3:
                    if i == a[j][0]:
                        out += 'o--'
                    elif i == a[j][1]:
                        out += 'x--'

                if b[2*i][j] == 3.5:
                    if i == a[j][0]:
                        out += 'o--'
                    elif i == a[j][1]:
                        out += a[j][-2][1] + '--'

                if b[2*i][j] == 4:
                    if i == a[j][0] or i == a[j][1]:
                        out += 'o--'
                    elif i == a[j][2]:
                        out += 'x--'


                if b[2*i+1][j] == 2:
                    space += '|  '
                if b[2*i+1][j] == 0:
                    space += '   '

            string_out[2*i].append(out+'-M')
            string_out[2*i+1].append(space+'  ')

        for i in string_out:
            print(i[0]+i[1])