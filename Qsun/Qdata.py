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
    '''Return the estimated Kolmogorov complexity of a dataset'''
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


