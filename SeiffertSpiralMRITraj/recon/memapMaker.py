#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:53:47 2024

@author: janastri
"""
import numpy as np 
import gc 

def makeMemMap(matrix, filename = "matrix_memap"):
    dtype= matrix.dtype
    shape=matrix.shape
    matrix_memmap = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
    matrix_memmap[:] = matrix[:]
    matrix_memmap.flush()

    del matrix
    matrix = np.memmap(filename, dtype=dtype, mode='r+', shape=shape)
    del matrix_memmap

    gc.collect()
    return matrix