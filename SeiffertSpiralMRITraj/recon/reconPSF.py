#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:19:27 2024

@author: janastri
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "19" # to make sure numpy uses multiple threads

import sys
sys.path.append('/export02/data/Jana/bart_cuda/bart-0.9.00/python')
from cfl import readcfl, writecfl
import sigpy as sp
from sigpy.mri.app import JsenseRecon

import sys

#import bart
import subprocess
import gc
#from ._extensions._pywt import DiscreteContinuousWavelet, Modes
#from ._extensions._dwt import dwt_max_level, dwt_axis

import cupy as cp
import numpy as np
from sigpy import backend, linop, prox, util, fourier, interp, thresh, wavelet
from math import ceil
#import pywt
#from reconForYB import kspace_precond # delete later and implement here 
import matplotlib.pyplot as plt

#from memapMaker import makeMemMap
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')

import nibabel as nib
def save_as_nifti(data, filename):
    # Create a NIfTI image object
    nifti_image = nib.Nifti1Image(data, np.eye(4))  # Assuming identity affine for simplicity

    # Save the NIfTI image to a file
    nib.save(nifti_image, filename)
    return

def _apodize(input, ndim, oversamp, width, beta):
    xp = backend.get_array_module(input)
    output = input
    for a in range(-ndim, 0):
        i = output.shape[a]
        os_i = ceil(oversamp * i)
        idx = xp.arange(i, dtype=output.dtype)

        # Calculate apodization
        apod = (beta**2 - (np.pi * width * (idx - i // 2) / os_i)**2)**0.5
        apod /= xp.sinh(apod)
        output *= apod.reshape([i] + [1] * (-a - 1))
    return output

def _scale_coord(coord, shape, oversamp):
    ndim = coord.shape[-1]
    #output = coord.copy() # bad bc adds memory
    output = coord # will modify it
    for i in range(-ndim, 0):
        scale = ceil(oversamp * shape[i]) / shape[i]
        shift = ceil(oversamp * shape[i]) // 2
        output[..., i] *= scale
        output[..., i] += shift
    return output


def kspace_precond_BART(mps, coord=None,
                   lamda=0, device=sp.Device(0),
                   oversamp=1.25, BART=0):
    r"""Compute a diagonal preconditioner in k-space.

    Considers the optimization problem:

    .. math::
        \min_P \| P A A^H - I \|_F^2

    where A is the Sense operator.

    Args:
        mps (array): sensitivity maps of shape [num_coils] + image shape.
        weights (array): k-space weights.
        coord (array): k-space coordinates of shape [...] + [ndim].
        lamda (float): regularization.

    Returns:
        array: k-space preconditioner of same shape as k-space.

    """
    dtype = mps.dtype

    #device = sp.Device(device)
    #xp = device.xp
    xp = np
    
    #coord = sp.to_device(coord) # on CPU
    # added
    #import numpy as np
    
    mps_shape = list(mps.shape)
    img_shape = mps_shape[1:]
    img2_shape = [i * 2 for i in img_shape]
    ndim = len(img_shape)

    scale = sp.prod(img_shape)**1.5 / sp.prod(img_shape)
    
    coordLen = coord.shape[:-1]
    
    with device:
        coord2 = coord * 2 # doesn't add new memory
        
        # make memmap 
        ones = xp.ones(coordLen, dtype=dtype)
        ones = np.reshape(ones,[1,-1])
        #ones = makeMemMap(ones, "ones_memmap")
        #gc.collect()
        
        # calculate PSF
        if BART: # adjoint nufft
            ones = sp.to_device(ones) # on CPU
            ksp_name = 'kspace2'
            image_name = 'image2'
            kloc_name = 'kloc2'
            writecfl(kloc_name, coord2.T)
            writecfl(ksp_name, np.expand_dims(ones.T, axis=[0,2]))
            psf =  nufft_adjoint_BART(ones, img2_shape, ksp_name=ksp_name, image_name=image_name, kloc_name=kloc_name)
        else:
            #psf = sp.nufft_adjoint(ones, coord2, img2_shape, oversamp=oversamp) # adds memory
            psf = nufft_adjoint(ones, coord, oshape=img_shape ,oversamp=oversamp) # adds memory
        
        # free memory
        del ones
        '''
        # delete memory map
        if os.path.exists("ones_memmap"):
            os.remove("ones_memmap")
        '''
        #p_inv = []
        # store in memmap
        p_inv_memmap = np.memmap("p_inv_memap", dtype=np.csingle, mode='w+', shape=(mps_shape[0], coordLen[0]))
        
        count = 0
        for mps_i in mps: # choose current sensitivity map [Si]
            #print(f'\rmap: {count}/{mps_shape[0]}', end='')
            xcorr_fourier = 0
            
            # calculate the correlation between all the sansitivity maps by multiplying them in the frequency domain
            mps_i = sp.to_device(mps_i, device) # on GPU
            
            for mps_j in mps: # Xcorr = sum_i,j( | fft( Si * Sj^H ) | )
                mps_j = sp.to_device(mps_j, device)
                
                xcorr_fourier += cp.abs(sp.fft(mps_i * cp.conj(mps_j), img_shape))**2
            
            del mps_j
            mps_i = sp.to_device(mps_i)
            
            # bring it back to the space domain
            xcorr = sp.ifft(xcorr_fourier)
            
            # free memort 
            del xcorr_fourier
            
            # multiply it with the PSF
            psf = sp.to_device(psf, device)
            xcorr *= psf
            psf = sp.to_device(psf)
            xcorr = sp.to_device(xcorr) # on CPU
            
            if BART: # forward nufft
                #p_inv_i = nufft_BART(xcorr, ksp_name=ksp_name, image_name=image_name, kloc_name=kloc_name)
                p_inv_i = nufft(xcorr, coord2, oversamp=oversamp) # adds memory
            else:
                #p_inv_i = sp.nufft(xcorr, coord2, oversamp=oversamp) # adds memory
                p_inv_i = nufft(xcorr, coord2, oversamp=oversamp) # adds memory
                
            # free memory 
            del xcorr
            cp._default_memory_pool.free_all_blocks()

            mps_i_norm2 = np.linalg.norm(mps_i)**2
            #p_inv.append(p_inv_i * scale / mps_i_norm2)
            p_inv_i = p_inv_i * scale / mps_i_norm2
            
            p_inv_i = (xp.abs(p_inv_i) + lamda) / (1 + lamda)
            
            p_inv_i[p_inv_i == 0] = 1 # problem is here
            p_inv_i = 1 / p_inv_i
            
            # add to memory map
            p_inv_memmap[count, :] = p_inv_i
            p_inv_memmap.flush()
            count+=1
            
        del psf
        #del coord2
        '''
        p = []
        for i in range(len(p_inv)):
            mtx = p_inv[i]
            
            p2 = (xp.abs(mtx) + lamda) / (1 + lamda)
            
            #plt.plot(p2)
            #plt.show()
            
            p2[p2 == 0] = 1 # problem is here
            p2 = 1 / p2
            
            #plt.plot(p2)
            # plt.show()
            
            p.append(p2)
            
            # free memory
            p_inv[i] = None

        p = np.stack(p, axis=0)
        '''
        #p_inv = (xp.abs(xp.stack(p_inv, axis=0)) + lamda) / (1 + lamda)
        #p_inv[p_inv == 0] = 1
        #p = 1 / p_inv
        #return p.astype(dtype)
        p_inv_memmap = np.memmap("p_inv_memap", dtype=np.csingle, mode='r+', shape=(mps_shape[0], coordLen[0]))
        return p_inv_memmap
    
def nufft(input, coord, oversamp=1.25, width=4, device=sp.Device(0)): # coord should be on CPU
    ndim = coord.shape[-1]
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5
    #os_shape = _get_oversamp_shape(input.shape, ndim, oversamp)
    input_shape = input.shape
    
    os_shape = list(input_shape)[:-ndim] + [ceil(oversamp * i) for i in input_shape[-ndim:]]

    #output = input.copy() # bad bc adds memory
    output = input 
    
    # Apodize
    _apodize(output, ndim, oversamp, width, beta)

    # Zero-pad
    output /= util.prod(input_shape[-ndim:])**0.5
    output = util.resize(output, os_shape)

    output = sp.to_device(output, device)
    # FFT
    output = fourier.fft(output, axes=range(-ndim, 0), norm=None)

    # Interpolate
    # load coord here for super low mem
    coord = sp.to_device(coord.copy(), device) # make a copy of coord on GPU
    
    coord_scaled = _scale_coord(coord.copy(), input_shape, oversamp)
    
    output = interp.interpolate(output, coord_scaled, kernel='kaiser_bessel', width=width, param=beta)
    
    del coord_scaled
    output = sp.to_device(output) # move output to CPU
    
    output /= width**ndim

    return output 

def nufft_BART(image, iterations = 1, image_name = 'image', kloc_name = 'kloc', ksp_name='kspace'): # image must be on CPU
    if image.ndim > 3:
        image = np.transpose(image, [1,2,3,0])
    writecfl(image_name, image)
    
    N = image.shape 
    
    # try gpu first 
    bash_command_GPU = f"bart nufft -x {N[0]}:{N[1]}:{N[2]} -w4 -o1.25 -g {kloc_name} {image_name} {ksp_name}"
    #bash_command_GPU = f"bart nufft -i -l {L2RegParam} -m {iterations} -g {kloc_name} {kspacename} {outputName}"
    
    bash_command_CPU = f"bart nufft -x {N[0]}:{N[1]}:{N[2]} -w4 -o1.25 {kloc_name} {image_name} {ksp_name}"
    
    try:
        out = subprocess.run(bash_command_GPU, shell=True, capture_output=True, text=True)
        out.check_returncode()  # This will raise a CalledProcessError if the return code is non-zero
    except Exception as e:
        # Catch all exceptions
        print("trying NUFFT without GPU...")
        out = subprocess.run(bash_command_CPU, shell=True, capture_output=True, text=True)
        if out.returncode != 0:
            error_message = out.stderr.strip()
            raise Exception(f"Error executing bash command: {error_message}")
    kspace = np.squeeze(np.array(readcfl(ksp_name))).T
    return kspace 

def nufft_adjoint_BART(ksp, N, nCoils = 1, kloc_name = 'kloc', ksp_name = 'kspace', image_name = 'image', L2RegParam = 0.1):
    # recon image with nufft 
    #writecfl(kloc_name, coord) # only need to do this once 
    writecfl(ksp_name, np.expand_dims(ksp.T, axis=[0,2]))
    
    # try gpu first 
    bash_command_GPU = f"bart nufft -a -x {N[0]}:{N[1]}:{N[2]} -w4 -o1.25 -g {kloc_name} {ksp_name} {image_name}"
    #bash_command_GPU = f"bart nufft -i -l {L2RegParam} -m {iterations} -g {kloc_name} {kspacename} {outputName}"
    
    bash_command_CPU = f"bart nufft -a -x {N[0]}:{N[1]}:{N[2]} -w4 -o1.25 {kloc_name} {ksp_name} {image_name}"
    try:
        out = subprocess.run(bash_command_GPU, shell=True, capture_output=True, text=True)
        out.check_returncode()  # This will raise a CalledProcessError if the return code is non-zero
    except Exception as e:
        # Catch all exceptions
        print("trying NUFFT without GPU...")
        out = subprocess.run(bash_command_CPU, shell=True, capture_output=True, text=True)
        if out.returncode != 0:
            error_message = out.stderr.strip()
            raise Exception(f"Error executing bash command: {error_message}")
            
    if nCoils > 1:
        # root mean square of all images (along coil dim)
        #!bart rss $(bart bitmask 3) phantom_img phantom_img_rss
        image_name_rss = image_name + "_rss"
        print("calculating rss...")
        bash_command = f"bart rss $(bart bitmask 3) {image_name} {image_name_rss}"
        out = subprocess.run(bash_command, shell=True, capture_output=True, text=True)
        if out.returncode != 0:
            error_message = out.stderr.strip()
            raise Exception(f"Error executing bash command: {error_message}")
        
        img = readcfl(image_name_rss)
    else:
        img = readcfl(image_name)
        
        if img.ndim > 3:
            img = np.transpose(img, [3, 0, 1, 2])
    return img

def nufft_adjoint(input, coord, oshape=None, oversamp=1.25, width=4, device=sp.Device(0)): # coord should be on CPU
    ndim = coord.shape[-1] # 3
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5 # 6.996...
    if oshape is None:
        oshape = list(input.shape[:-coord.ndim + 1]) + fourier.estimate_shape(coord)
    else:
        oshape = list(oshape)

    #os_shape = _get_oversamp_shape(oshape, ndim, oversamp)
    os_shape = list(oshape)[:-ndim] + [ceil(oversamp * i) for i in oshape[-ndim:]]
    
    # Gridding
    #coord_GPU = sp.to_device(coord.copy(), device) # make a copy of coord on GPU
    
    coord =  sp.to_device(coord, device)
    
    coord_scaled = _scale_coord(coord.copy(), oshape, oversamp)
    input = sp.to_device(input, device)
    output = interp.gridding(input, coord_scaled, os_shape, kernel='kaiser_bessel', width=width, param=beta) # this part needs GPU or else really slow
    
    #coord =  sp.to_device(coord) # to CPU
    # free 
    del coord_scaled
    
    output /= width**ndim

    # IFFT
    output = fourier.ifft(output, axes=range(-ndim, 0), norm=None) # this is okay without GPU (even for large data size?)

    # Crop
    output = util.resize(output, oshape)
    output *= util.prod(os_shape[-ndim:]) / util.prod(oshape[-ndim:])**0.5

    # Apodize
    _apodize(output, ndim, oversamp, width, beta)
    output = sp.to_device(output) # move output to CPU
    
    return output

def forwardBART(x, mps, coord, device=sp.Device(0)): # needs to be done with BART's NUFFT
    with device:
        mps = sp.to_device(mps, device) # on GPU [C, 160, 160, 160]
        v = sp.to_device(x, device) # move x to device
        
        v = v * mps # apply linear operator [160, 160, 160] * [C, 160, 160, 160] -->  [C, 160, 160, 160]
    # free memory
    mps = sp.to_device(mps) # transfer back to CPU
    v = sp.to_device(v)
    
    # apply NUFFT 
    #coord = sp.to_device(coord, device) # on GPU [M, 3]
    # keep coord on CPU as nufft will make a copy to the GPU 
    # x is image, and using coordinates, you get the YB's frequency guess (can also use BART's NUFFT)
    v = nufft_BART(v) # nufft [8, 160, 160, 160] -->  [1, M]
    # free memory
    return v

def forwardSeperateCoils(x, mps, coord, forwardOutShape, C, device=sp.Device(0)):
    # vector to collect output
    #output = np.empty(forwardOutShape, dtype=x.dtype)
    output = np.memmap("output_memap", dtype=x.dtype, mode='w+', shape=forwardOutShape)
    
    for c in range(C):
        #print(f'\r Forward: {c}/{forwardOutShape[0]}')
        # apply sensitiviy map operator 
        mps_c = mps[ c, : , :, :] # select only one sensitivty map at a time 
        with device:
            mps_c = sp.to_device(mps_c, device) # on GPU [1, 160, 160, 160]
            v = sp.to_device(x, device) # move x to device
            
            v = v * mps_c # apply linear operator [160, 160, 160] * [1, 160, 160, 160] -->  [1, 160, 160, 160]
        # free memory
        del mps_c 
        
        # apply NUFFT 
        with device:
            #coord = sp.to_device(coord, device) # on GPU [M, 3] 
            # keep coord on CPU as nufft will make a copy to the GPU 
            # x is image, and using coordinates, you get the YB's frequency guess (can also use BART's NUFFT)
            v = nufft(v, coord, oversamp=1.25, width=4, device=device) # nufft [1, 160, 160, 160] -->  [1, M]
        # free memory
        #coord = sp.to_device(coord) # transfer back to CPU
        #v = sp.to_device(v) 
        
        # add collecting input into output vector 
        slc = tuple([slice(None)] * 0 + [slice(c, c+1)] + [slice(None)] * (2 - 0 - 1))
        output[slc] = v
        #output[c,:] = v
        #output.flush()
    output = np.memmap("output_memap", dtype=x.dtype, mode='r+', shape=forwardOutShape)
    return output

def backwardBART(v, mps, coord, N, C, device=sp.Device(0)):
    '''
    v -> [C, M] k-space domain information 
    N -> [N, N, N]
    '''
    output = 0 # initialized to scalar, but will become [N,N,N] vector
    
    v = nufft_adjoint_BART(v, N) # [1, M] ---> [1, N, N, N]
    
    mps = sp.to_device(mps, device) # on GPU [1, 160, 160, 160]
    xp = device.xp
    # take complex conjugate of it 
    mps = xp.conj(mps)
    
    v = sp.to_device(v, device) # move to GPU
    
    v = v * mps # apply linear operator [1, 160, 160, 160] * [1, 160, 160, 160] -->  [1, 160, 160, 160]
    
    v = xp.squeeze(xp.sum(v, axis=0))

    v = sp.to_device(v) # move to CPU 

    return v

def backwardSeperateCoils(v, mps, coord, N, C, device=sp.Device(0)):
    '''
    v -> [C, M] k-space domain information 
    N -> [N, N, N]
    '''
    output = 0 # initialized to scalar, but will become [N,N,N] vector
    for c in range(C):
        v_c = v[c,:] # only one coil's info [1, M]
        v_c = np.reshape(v_c, [1,-1])
        
        # apply nufft adjoint 
        with device:
            v_c = sp.to_device(v_c, device) # move to GPU
            #coord = sp.to_device(coord, device)  # move to GPU
            # keep coord on CPU as nufft will make a copy to the GPU 
            oshape = [1]
            oshape.extend(N)
            
            v_c = nufft_adjoint(v_c, coord, oshape, oversamp=1.25, width=4, device=device) # [1, M] ---> [1, N, N, N]
            
        # multiply by mps ^ H
        mps_c = mps[ c, : , :, :] # select only one sensitivty map at a time 
        with device:
            mps_c = sp.to_device(mps_c, device) # on GPU [1, 160, 160, 160]
            xp = device.xp
            
            # take complex conjugate of it 
            mps_c = xp.conj(mps_c)
            
            v_c = sp.to_device(v_c, device) # move to GPU
            
            v_c = v_c * mps_c # apply linear operator [1, 160, 160, 160] * [1, 160, 160, 160] -->  [1, 160, 160, 160]
        del mps_c
        v_c = sp.to_device(v_c) # move to CPU 
        
        # sum  (this one does nothing)
        #with device:
        #    v_c = xp.sum(v_c, axis=())
        
        # reshape  [1, 160, 160, 160] -->  [160, 160, 160] aka to N
        v_c = v_c.reshape(N)
        
        output += v_c # sum all coil images in [N,N,N] array 
    return output
    
def multPrecond(v, precond, device=sp.Device(0)):
    v = sp.to_device(v, device) # move to GPU
    precond = sp.to_device(precond, device)
    
    v = v *  precond # apply linear operator [8, M] * [8, M] -->  [8, M]
    
    v = sp.to_device(v) # move back to cpu
    precond = sp.to_device(precond)
    return v

def findMaxEigen(mps, coord, precond, dtype, forwardOutShape, device=sp.Device(0), eigen_iter = 30, BART = 0):
    N = mps.shape[1:]
    C = mps.shape[0] # apply coilSens operator one map at a time 
    
    # make inital x image 
    x = util.randn(N, dtype=dtype, device=sp.cpu_device) # on CPU [160, 160, 160]
    
    print(f'\rProgress: 0/{eigen_iter}', end='')
    
    eigen_all =[]
    
    for i in range(eigen_iter):
        # apply foward (move to k-space domain)
        if BART: # forward nufft
            #v = forwardBART(x, mps, coord, device=device)
            v = forwardSeperateCoils(x, mps, coord, forwardOutShape, C, device=device) # -->  [8, M] ADDS MEMORY
        else:
            v = forwardSeperateCoils(x, mps, coord, forwardOutShape, C, device=device) # -->  [8, M] ADDS MEMORY
        #v2 = forward(x, mps, coord, device)
        
        # multiply with preconditioner
        #v = multPrecond(v, precond, device=device) # [8, M] * [8, M] -->  [8, M] on GPU
        #v = v *  precond # elementwise multiplication on CPU
        v *= precond
        v.flush()
        
        if BART: # adjoint nufft
            v = backwardBART(v, mps, coord, N, C, device=device) # [8, M] --> [N,N,N]
        else:
            # apply backward model (back in image domain)
            v = backwardSeperateCoils(v, mps, coord, N, C, device=device) # [8, M] --> [N,N,N]
            
        # calc max eigenvalue
        max_eig = np.linalg.norm(v).item() # matrix L2 norm of image
        
        # scale matrix
        v = v / max_eig
        
        # copy contents of v to x for next round
        np.copyto(x, v)
        # or x = v
        
        # free space
        del v
            
        eigen_all.append(max_eig)
        print(f'\rProgress: {i+1}/{eigen_iter}', end='')
        
    if os.path.exists("output_memap"):
        os.remove("output_memap")
        
    return max_eig, eigen_all

def l1WaveletRecon(image_guess):
    N = image_guess.shape
    w = wavelet.fwt(image_guess, wave_name='db4', axes=None, level=None)
    return w

def  l1WaveletRecon_H(k_guess, N): # device of input doesn't matter, will be copied to CPU
    _, coeff = wavelet.get_wavelet_shape(N, wave_name='db4', axes=None, level=None)

    return wavelet.iwt(k_guess, N, coeff, wave_name='db4', axes=None, level=None)

def calcObjective(ksp, image_guess, mps, coord, lamda = 0.01, BART=0, device=sp.Device(0)): # all should be in numpy arrays
    if BART: # forward nufft
        #ksp_guess = forwardBART(image_guess, mps, coord, device=device)
        ksp_guess = forwardSeperateCoils(image_guess, mps, coord, ksp.shape, ksp.shape[0], device=device)
    else:
        ksp_guess = forwardSeperateCoils(image_guess, mps, coord, ksp.shape, ksp.shape[0], device=device)
        
    r = ksp_guess - ksp # error 
    
    obj = ( 1 / 2 ) * ( np.linalg.norm(r).item()**2 )
    
    # plus add wavelet regularization to the objective function 
    obj += lamda * np.sum(np.abs(l1WaveletRecon(image_guess))).item()
    return obj

def PD_alg(coord, ksp, precond, mps, max_eig, iterSave = [], max_iter = 100, BART=0, savename='', continuation=0, device=sp.Device(0)):
    # image reconstruction with the partial dual algorithm 
    # make initial guesses
    C =  ksp.shape[0]
    kshape = ksp.shape
    N = mps.shape[1:]
    
    obj_all = [] # keep track of objective function 
    
    gamma_primal = 0 
    gamma_dual = 1
    
    tol = 0
    
    if continuation:
        image_guess = np.save('image_guess')
        image_ext= np.save('image_ext')
        obj_all = np.save('obj_all')
        obj_all = list(obj_all)
        u = np.save('u')
        theta = np.save('theta')
        precond = np.save('precond')
        sigma_min = np.save('sigma_min')
        tau = np.save('tau')
    else:
        tau = 1 / max_eig
        
        #u = np.zeros(kshape, dtype=ksp.dtype) # [C, M] vector in k-domain
        
        # make mem map instead 
        u = np.memmap("u_memap", dtype=ksp.dtype, mode='w+', shape=kshape)
        u[:] = 0 + 0j
        u.flush()
        
        image_guess = np.zeros(N, dtype=ksp.dtype) # [N, N, N] vector in image-domain
        
        sigma_min = np.min(np.abs(precond)).item() # absolute min value of the preconditioner 
        #resid = np.infty # we will try to make this into a small number, so initalize to big
        
        image_ext = image_guess
        
        print(f'\rProgress: {0}/{max_iter}', end='')
    
    # start primal-dual 
    for i in range(max_iter):
        
        if continuation: # continuation index
            i = i + len(obj_all)
            
        # update dual 
        # u = [precond] * A(image_guess) + u
        if BART: # forward nufft
            #ksp_guess = forwardBART(image_ext, mps, coord, device=device)
            ksp_guess = forwardSeperateCoils(image_ext, mps, coord, ksp.shape, ksp.shape[0], device=device)
        else:
            ksp_guess = forwardSeperateCoils(image_ext, mps, coord, ksp.shape, ksp.shape[0], device=device) # outputs a memory map
        
        '''
        # back to dual 
        u += precond * ksp_guess # check if you can accelerate with GPU (might need to do in chunks, takes long )
        u.flush()
        
        # free
        del ksp_guess
        if os.path.exists("output_memap"):
            os.remove("output_memap")
            
        # perform L2Reg with true ksp and u ( u = (u - precond * ksp) / (1 + precond) )
        u += precond * (-ksp) # check if you can accelerate with GPU (memory error)
        u.flush()
        
        u /= 1 + precond # u is now the error in the k-domain between the guess and real ksp measurements
        u.flush()
        # didn't use copyto here to transfer to u 
        '''
        # perform in batches
        chunk_size = 10000000
        for j in range(0, precond.shape[1], chunk_size):
            j_end = min(j + chunk_size, precond.shape[1])
            
            # back to dual 
            u[:, j:j_end] += precond[:, j:j_end] * ksp_guess[:, j:j_end]
            
            # perform L2Reg with true ksp and u ( u = (u - precond * ksp) / (1 + precond) )
            u[:, j:j_end] += precond[:, j:j_end] * (-ksp[:, j:j_end])
            
            u[:, j:j_end] /= 1 + precond[:, j:j_end]
            
            # Optionally, flush the chunk to disk immediately
            u.flush()
        
        del ksp_guess
        if os.path.exists("output_memap"):
            os.remove("output_memap")
            
        # update primal ( image_guess = image_guess - tau * A^H (u) )
        image_old = image_guess.copy() # keep track of old image 
        
        if BART: # backwards nufft
            image_err = backwardBART(u, mps, coord, N, C, device=device) # [8, M] --> [N,N,N]
        else:
            # apply backward model (back in image domain)
            image_err = backwardSeperateCoils(u, mps, coord, N, C, device=device)
        
        image_guess += -tau * image_err
        del image_err
        
        # L1 reg [ wavelet^H ( L1( tau, wavelet( image_guess ) ))] = [ wavelet^H ( L1_err_k-domain ))] = L1_err_image_domain
        W_img = l1WaveletRecon(image_guess) # in k-space now
        del image_guess
        
        # move to GPU
        W_img = sp.to_device(W_img, device)
        k_err = thresh.soft_thresh( 0.01*tau, W_img) # can do this on GPU with C kernel (or cpu)
        k_err = sp.to_device(k_err) # back to CPU
        del W_img
        
        # inverse wavelet transform 
        #image_err = l1WaveletRecon_H(k_err, kshape)
        image_guess = l1WaveletRecon_H(k_err, N) # wavelet only works on CPU
        
        # update preconditioner 
        theta = 1 / (1 + 2 * gamma_dual * sigma_min)**0.5
        precond *= theta
        sigma_min *= theta
        tau /= theta
        
        # Extrapolate primal
        image_diff = image_guess - image_old
        #resid = np.linalg.norm(image_diff / tau**0.5).item()
        
        image_ext = image_guess + theta * image_diff # i think this won't add memory 
        
        if i%1 == 0: # only calculate objective every second time (to speed up)
            # calc object function
            obj_all.append(calcObjective(ksp, image_guess, mps, coord, BART=BART))
            
        print(f'\rProgress: {i+1}/{max_iter}', end='')
        
        if i+1 in iterSave:
            save_as_nifti(abs(image_guess), savename + "_" + str(i+1) +"iter.nii")
            
            fig, axes = plt.subplots(1, 3)
            axes[0].imshow(abs(image_guess[:,:,int(image_guess.shape[2]/2)]), cmap="grey", interpolation = 'none')
            axes[0].axis('off') 
            axes[1].imshow(abs(image_guess[:,int(image_guess.shape[1]/2),:]), cmap="grey", interpolation = 'none')
            axes[1].axis('off') 
            axes[2].imshow(abs(image_guess[int(image_guess.shape[0]/2),:,:]), cmap="grey", interpolation = 'none')
            axes[2].axis('off') 
            titleName = 'L1 wavelet PDHG MC precond'
            plt.suptitle(titleName)
            plt.tight_layout()
            plt.savefig(savename + "_" + str(i+1) + "iter.svg", format = 'svg')
            plt.show()
            
            np.save('image_guess', image_guess)
            np.save('image_ext', image_ext)
            np.save('obj_all', np.array(obj_all))
            np.save('u', u)
            np.save('theta', theta)
            np.save('precond', precond)
            np.save('sigma_min', sigma_min)
            np.save('tau', tau)
            
    return image_guess, obj_all

# function summarizing what's happening 
def performRecon(ksp, coord , iterSave=[], max_iter=50, BART=0, saveName='', continuation=0, device=sp.Device(0)): # all inputs should be on CPU
    '''
    ksp = k-space measurements [8, M]
    coord = k-space coordinattes [M, 3]
    mps = sensitivity maps [C, N, N, N]
    precond = previously computed preconditioning array [8, N]
    '''
    if ksp.ndim == 1:
        ksp = np.reshape(ksp, [ksp.shape[0],1]).T
        
    print(cp.get_default_memory_pool().used_bytes())
    
    print('calculating maps...', end = ' ')
    mps = JsenseRecon(ksp, coord=coord, device=sp.Device(0)).run()
    print('done!')
    print(str(mps.shape))
    
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()
    print(cp.get_default_memory_pool().used_bytes())
    
    # make mps a memeroy map 
    #mps = makeMemMap(mps, "mps_memmap")
    gc.collect()
    
    
    #delete this 
    #mps = np.memmap("mps_memmap", dtype=np.csingle, mode='r+', shape= mps.shape)
    
    #mps = sp.to_device(mps, sp.Device(0))
    #coord = sp.to_device(coord, sp.Device(0)) # move to GPU for this function 
    print('calculating preconditioning matrix...', end = ' ')
    precond = kspace_precond_BART(mps, coord=coord, BART=BART, device=sp.Device(0))
    print('done!')
    
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    print(cp.get_default_memory_pool().used_bytes())
    #mps = sp.to_device(mps) # move back to CPU
    #coord = sp.to_device(coord)
    
    if BART:  # if we are using BART for nufft, need to write trajectory's coordinates 
        writecfl('kloc', coord.T)
        
    # step 1: calculate max eigen values of operator A.H * S * A, where A is the sense forward operator (NUFFT * coilSens) and S is the preconditioning matrix
    print("calculating eigen value...")
    max_eig, e_all = findMaxEigen(mps, coord, precond, ksp.dtype,  ksp.shape, eigen_iter = 6, BART=BART, device=device)
    print("done! Max Eigen-value = ", max_eig)
    
    plt.plot(list(range(1, len(e_all))), e_all[1:])
    plt.title("Eigen Values Approx of Sense")
    plt.xlabel("epoch")
    plt.ylabel("Eigen Value")
    plt.show()
    
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()
    print(cp.get_default_memory_pool().used_bytes())
    
    '''
    # delete this 
    mps = np.memmap("matrix_recon", dtype=np.csingle, mode='r+', shape= (32, 300, 300, 300))
    precond = np.memmap("p_inv_memap", dtype=np.csingle, mode='r+', shape=(32, ksp.shape[1]))
    max_eig = 1181.3935546875
    '''
    
    print("starting image recon with Partial-Dual algorithm...")
    # step 2: start primal dual algorithm
    image, obj_values = PD_alg(coord, ksp, precond, mps, max_eig, BART=BART, max_iter=max_iter, iterSave=iterSave, savename=saveName, continuation=continuation, device=device)
    print("done!")
    
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()
    print(cp.get_default_memory_pool().used_bytes())
    
    plt.plot(list(range(1, len(obj_values))), obj_values[1:])
    plt.title("Partial-Dual Alg Error")
    plt.xlabel("epoch")
    plt.ylabel("Objective Function")
    plt.show()
    
    plt.plot(list(range(int(len(obj_values)*1/2), len(obj_values))), obj_values[int(len(obj_values)*1/2):])
    plt.title("Partial-Dual Alg Error (only end)")
    plt.xlabel("epoch")
    plt.ylabel("Objective Function")
    plt.show()
    
    return image, obj_values # return image and objective func scores 

if __name__ == "__main__":
    '''
    # input files
    #ksp_file = '/export02/data/Jana/NewSpiral/YB_1Echos_240FOV_4msRO_1p5mmRes_0p5alpha_SpiralInOut/R2/sim/phantomKspace_8Coils'
    #coord_file = '/export02/data/Jana/NewSpiral/YB_1Echos_240FOV_4msRO_1p5mmRes_0p5alpha_SpiralInOut/R2/YB_1Echos_240FOV_4msRO_1p5mmRes_0p5alpha_SpiralInOut_R2_allInterleaves_nyTest.npy'
    
    #ksp_file = '/export02/data/Jana/NewSpiral/YB_1Echos_240FOV_4msRO_1p5mmRes_0p5alpha_SpiralInOut/R1/sim/phantomKspace_8Coils'
    #coord_file = '/export02/data/Jana/NewSpiral/YB_1Echos_240FOV_4msRO_1p5mmRes_0p5alpha_SpiralInOut/R1/YB_1Echos_240FOV_4msRO_1p5mmRes_0p5alpha_SpiralInOut_R1_allInterleaves_nyTest.npy'
    
    #ksp_file = '/export02/data/Jana/NewSpiral/YB_1Echos_240FOV_4msRO_1p5mmRes_1alpha_SpiralInOut/R1/sim/phantomKspace_8Coils'
    #coord_file = '/export02/data/Jana/NewSpiral/YB_1Echos_240FOV_4msRO_1p5mmRes_1alpha_SpiralInOut/R1/YB_1Echos_240FOV_4msRO_1p5mmRes_1alpha_SpiralInOut_R1_allInterleaves_nyTest.npy'

    #ksp_file = '/export02/data/Jana/NewSpiral/YB_1Echos_240FOV_4msRO_1p5mmRes_1alpha_SpiralInOut/R6/sim/phantomKspace_8Coils'
    #coord_file = '/export02/data/Jana/NewSpiral/YB_1Echos_240FOV_4msRO_1p5mmRes_1alpha_SpiralInOut/R6/YB_1Echos_240FOV_4msRO_1p5mmRes_1alpha_SpiralInOut_R6_allInterleaves_nyTest.npy'
    
    ksp_file = '/export02/data/Jana/NewSpiral/YB_1Echos_240FOV_4msRO_1p5mmRes_0p5alpha_SpiralInOut/Rshots1000/sim/phantomKspace_8Coils'
    kloc_name = '/export02/data/Jana/NewSpiral/YB_1Echos_240FOV_4msRO_1p5mmRes_0p5alpha_SpiralInOut/Rshots1000/YB_1Echos_240FOV_4msRO_1p5mmRes_0p5alpha_SpiralInOut_allInterleaves.npy'
    

    # formating data
    print('loading ksp and coord data...')
    FOV = 0.24
    ksp = np.squeeze(np.array(readcfl(ksp_file))).T
    coord = np.load(kloc_name)
    coord = np.reshape(np.transpose(coord,[2,0,1]), [coord.shape[0]*coord.shape[2], coord.shape[1]])
    coord = coord * FOV
    coord = np.array(coord, dtype = np.float32)
    '''
    
    kloc_name = '/export02/data/Jana/NewSpiral/YB_1Echo_240fov_6msRO_0p8mm_1alpha/233_Jan25_yarn_Jana/traj/YB_1Echo_240fov_6msRO_0p8mm_1alpha_allInterleaves_nyTest_2.npy'
    ksp_file = '/export02/data/Jana/NewSpiral/YB_1Echo_240fov_6msRO_0p8mm_1alpha/233_Jan25_yarn_Jana/twix/meas_MID00071_FID19969_JS_yarn_matching.dat'
    
    coord = np.load(kloc_name) 
    npts, _, Nshots = coord.shape
    del coord 
    
    # make memroymap
    coord = np.memmap("coord_recon", dtype=np.float32, mode='r+', shape=(npts*Nshots, 3))
    ksp = np.memmap("ksp_recon", dtype=np.csingle, mode='r+', shape= (32, npts*Nshots))
    
    
    savename = ksp_file.split('.')[0] + "_imgPDHGL1Wavelet_MCprecon"
    

    pdhg_mc_img, obj_values = performRecon(ksp, coord, BART=0, max_iter=200, iterSave=[5, 10, 20, 30, 50, 75, 100, 125, 150, 175], saveName=savename, continuation=0, device=sp.Device(0))
    
    
    save_as_nifti(abs(pdhg_mc_img), savename + "_200iter.nii")
    
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(abs(pdhg_mc_img[:,:,int(pdhg_mc_img.shape[2]/2)]), cmap="grey", interpolation = 'none')
    axes[0].axis('off') 
    axes[1].imshow(abs(pdhg_mc_img[:,int(pdhg_mc_img.shape[1]/2),:]), cmap="grey", interpolation = 'none')
    axes[1].axis('off') 
    axes[2].imshow(abs(pdhg_mc_img[int(pdhg_mc_img.shape[0]/2),:,:]), cmap="grey", interpolation = 'none')
    axes[2].axis('off') 
    titleName = 'L1 wavelet PDHG MC precond'
    plt.suptitle(titleName)
    plt.tight_layout()
    plt.savefig( savename + "_200iter.svg", format = 'svg')
    plt.show()
