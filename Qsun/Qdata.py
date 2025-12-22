# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 11:30:23 2021

@author: ASUS
"""
import numpy as np
import math

def intrinsic_dim_from_cov(dataset):
    '''Return the estimated Intrinsic Dimension using the spectrum of the covariance matrix'''
    cov = np.cov(dataset, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    return ((eigvals.sum())**2) / ((eigvals**2).sum())

def sepctral_complex_kernel(kernel, lambda_K = 0.1):
    '''Return the Spectral complexity of the Kernel matrix'''
    return np.trace(kernel @ np.linalg.inv(kernel + lambda_K * np.identity(kernel.shape[0])))

def kolmogorov_complex(dataset):
    '''Return the estimated Kolmogorov complexity of the dataset'''
    import zlib, gzip, lzma, bz2
    
    def serialize_array(X):
        '''Return the raw data bytes of the dataset'''
        # Simple deterministic serialization: shape + dtype + raw bytes
        header = f"{X.shape}|{str(X.dtype)}|".encode('utf-8')
        body = X.tobytes(order='C')
        return header + body

    def compressed_size_bytes(data_bytes, alg='gzip'):
        '''Compress the data (bytes)'''
        # data_bytes: bytes
        if alg == 'zlib':
            out = zlib.compress(data_bytes)
        elif alg == 'gzip':
            out = gzip.compress(data_bytes)
        elif alg == 'lzma':
            out = lzma.compress(data_bytes)
        elif alg == 'bz2':
            out = bz2.compress(data_bytes)
        else:
            raise ValueError("Unknown alg")
        return len(out)
    
    b = serialize_array(dataset)
    sizes = {}
    algs=('zlib', 'gzip', 'lzma', 'bz2')
    for a in algs:
        sizes[a] = compressed_size_bytes(b, alg=a)
    sizes['original_bytes'] = len(b)
    sizes['best_bytes'] = min(sizes[a]/len(b) for a in algs)
    return sizes

"""

Updated on Saturday Dec 20th, 2025

"""
class PauliZExpectation:
    def __init__(self, wavefunction):
        self.wavefunction = wavefunction
        self.state = wavefunction.state 
        self.amplitudes = wavefunction.amplitude  
        self.probs = wavefunction.probabilities()  
        self.n_qubits = len(self.state[0])
        
    def _bit_at_position(self, bitstring, position):
        """
        Extract bit at position from bitstring
        bitstring: '0101'
        position: 1 -> returns '1'
        """
        return bitstring[position]

    def _count_ones(self, bitstring, positions):
        """
        Count number of '1' bits at given positions
        bitstring: '1101'
        positions: [0, 1, 2] -> returns 2
        """
        count = 0
        for pos in positions:
            if bitstring[pos] == '1':
                count += 1
        return count

    def _parity(self, count):
        """
        Check if count is even (return +1) or odd (return -1)
        """
        if count % 2 == 0:
            return +1
        else:
            return -1

    def _validate_indices(self, *indices):
        """
        Check if qubit indices are valid
        - 0 <= index < n_qubits
        - No duplicates
        """
        # Check duplicates
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate indices found: {indices}")
        
        # Check range
        for idx in indices:
            if idx < 0 or idx >= self.n_qubits:
                raise ValueError(f"Index {idx} out of range [0, {self.n_qubits-1}]")

    def one_body(self, i):
        self._validate_indices(i)
        probs_0 = 0.0
        probs_1 = 0.0
        for idx, state in enumerate(self.state):
            if state[i] == '0':
                probs_0 += self.probs[idx]
            else:
                probs_1 += self.probs[idx]
        return probs_0 - probs_1
    
    def two_body(self, i, j):
        self._validate_indices(i, j)
        probs_same = 0.0
        probs_diff = 0.0
        for idx, state in enumerate(self.state):
            if state[i] == state[j]:
                probs_same += self.probs[idx]
            else:
                probs_diff += self.probs[idx]
        return probs_same - probs_diff

    def three_body(self, i, j, k):
        self._validate_indices(i, j, k)
        probs_even = 0.0
        probs_odd = 0.0
        for idx, state in enumerate(self.state):
            count = self._count_ones(state, [i, j, k])
            if count % 2 == 0:  
                probs_even += self.probs[idx]
            else:  
                probs_odd += self.probs[idx]
        return probs_even - probs_odd
    
    def four_body(self, i, j, k, l):
        self._validate_indices(i, j, k, l)
        probs_even = 0.0
        probs_odd = 0.0
        for idx, state in enumerate(self.state):
            count = self._count_ones(state, [i, j, k, l])
            if count % 2 == 0: 
                probs_even += self.probs[idx]
            else: 
                probs_odd += self.probs[idx]
        return probs_even - probs_odd
    

class ConnectedCorrelator:
    def __init__(self, wavefunction):
        self.wavefunction = wavefunction
        self.exp_val = PauliZExpectation(wavefunction)

    def u2(self, i, j):
        two_body = self.exp_val.two_body(i, j)  # ⟨ZiZj⟩
        one_body_i = self.exp_val.one_body(i)   # ⟨Zi⟩
        one_body_j = self.exp_val.one_body(j)   # ⟨Zj⟩
        
        return two_body - one_body_i * one_body_j
    
    def u3(self, i, j, k):
        three_body = self.exp_val.three_body(i, j, k)
    
        two_ij = self.exp_val.two_body(i, j)
        two_ik = self.exp_val.two_body(i, k)
        two_jk = self.exp_val.two_body(j, k)
        
        one_i = self.exp_val.one_body(i)
        one_j = self.exp_val.one_body(j)
        one_k = self.exp_val.one_body(k)
        
        result = three_body \
                - two_ij * one_k \
                - two_ik * one_j \
                - two_jk * one_i \
                + 2 * one_i * one_j * one_k
        
        return result
    
    def u4(self, i, j, k, l):
        four_body = self.exp_val.four_body(i, j, k, l)
        
        term_31_1 = self.u3(i, j, k) * self.exp_val.one_body(l)
        term_31_2 = self.u3(i, j, l) * self.exp_val.one_body(k)
        term_31_3 = self.u3(i, k, l) * self.exp_val.one_body(j)
        term_31_4 = self.u3(j, k, l) * self.exp_val.one_body(i)
        
        term_22_1 = self.u2(i, j) * self.u2(k, l)
        term_22_2 = self.u2(i, k) * self.u2(j, l)
        term_22_3 = self.u2(i, l) * self.u2(j, k)
        
        one_i = self.exp_val.one_body(i)
        one_j = self.exp_val.one_body(j)
        one_k = self.exp_val.one_body(k)
        one_l = self.exp_val.one_body(l)
        
        term_211_1 = self.u2(i, j) * one_k * one_l
        term_211_2 = self.u2(i, k) * one_j * one_l
        term_211_3 = self.u2(i, l) * one_j * one_k
        term_211_4 = self.u2(j, k) * one_i * one_l
        term_211_5 = self.u2(j, l) * one_i * one_k
        term_211_6 = self.u2(k, l) * one_i * one_j
        
        term_comp = 3 * one_i * one_j * one_k * one_l
        
        result = four_body \
                - (term_31_1 + term_31_2 + term_31_3 + term_31_4) \
                - (term_22_1 + term_22_2 + term_22_3) \
                - (term_211_1 + term_211_2 + term_211_3 + 
                    term_211_4 + term_211_5 + term_211_6) \
                + term_comp
        
        return result
    
"""
Updated on Sunday Dec 21th, 2025

"""
class EntanglementEntropy:
    
    def __init__(self, wavefunction):
        self.wavefunction = wavefunction
        self.n_qubits = len(wavefunction.state[0])
    
    def reduced_density_matrix(self, keep_qubits):
        states = self.wavefunction.state
        amplitudes = self.wavefunction.amplitude
        
        if not all(0 <= q < self.n_qubits for q in keep_qubits):
            raise ValueError(f"keep_qubits must be in range [0, {self.n_qubits - 1}]")
        
        trace_out_qubits = [i for i in range(self.n_qubits) if i not in keep_qubits]
        dim_reduced = 2 ** len(keep_qubits)
        reduced_rho = np.zeros((dim_reduced, dim_reduced), dtype=complex)
        
        for i in range(len(states)):
            for j in range(len(states)):
                state_i = states[i]
                state_j = states[j]
                
                traced_match = all(state_i[q] == state_j[q] for q in trace_out_qubits)
                if not traced_match:
                    continue
                
                reduced_state_i = ''.join([state_i[q] for q in keep_qubits])
                reduced_state_j = ''.join([state_j[q] for q in keep_qubits])
                idx_i = int(reduced_state_i, 2)
                idx_j = int(reduced_state_j, 2)
                reduced_rho[idx_i, idx_j] += amplitudes[i] * np.conj(amplitudes[j])
        
        return reduced_rho
    
    def von_neumann_entropy(self, keep_qubits=None, base=2):
        if keep_qubits is None:
            rho = np.outer(self.wavefunction.amplitude, np.conj(self.wavefunction.amplitude))
        else:
            rho = self.reduced_density_matrix(keep_qubits)
        
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        
        if base == 2:
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        elif base == np.e:
            entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        else:
            entropy = -np.sum(eigenvalues * np.log(eigenvalues) / np.log(base))
        
        return entropy
    
    def entanglement_entropy(self, bipartition, base=2):
        if isinstance(bipartition, tuple) and len(bipartition) == 2:
            qubits_A, qubits_B = bipartition
            qubits_A = list(qubits_A)
        else:
            qubits_A = list(bipartition)
            qubits_B = [i for i in range(self.n_qubits) if i not in qubits_A]
        
        if len(qubits_A) + len(qubits_B) != self.n_qubits:
            raise ValueError("Bipartition must cover all qubits exactly once")
        if set(qubits_A) & set(qubits_B):
            raise ValueError("qubits_A and qubits_B must be disjoint")
        
        return self.von_neumann_entropy(keep_qubits=qubits_A, base=base)