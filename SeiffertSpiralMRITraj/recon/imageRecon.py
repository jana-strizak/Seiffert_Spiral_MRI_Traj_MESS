#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:19:27 2024

@author: janastri
"""
import os
#os.environ["OPENBLAS_NUM_THREADS"] = "19" # to make sure numpy uses multiple threads

import sigpy as sp
from sigpy.mri.app import JsenseRecon
import subprocess
import gc
import nibabel as nib
import cupy as cp
import numpy as np
from sigpy import backend, util, fourier, thresh, wavelet, interp
from math import ceil
import matplotlib.pyplot as plt

from SeiffertSpiralMRITraj.recon.memapMaker import makeMemMap
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')

import mapvbvd

def readMRIdataMESS(ksp_file, traj, echo = 0, R = None, Ncoils = 32):
    '''
    Extracting MRI twix data into a numpy array of kspace values
    Inputs:
        ksp_file : { string } containing the location of the twix file
        Npoints: { Int } number of samples
        R: { int } undersampling factor desired for retrospectivally undersampling the data
        Ncoils: { int } number of recieve channel coils of the MRI initalized to the Neuro's 7T containing 32 channels
    Outputs:
        ksp: MRI aquired signal in a numpy array of shape Ncoils by Npoints
    '''
    # getting the number of k-sapce samples 
    Npoints = traj.kspaceCoord.shape[0]
    # getting the echo borders 
    borders = traj.localmaxIdx.copy()
    borders[-1] = borders[-1] + 1
    
    # get file extention
    _, ext = os.path.splitext(ksp_file)
    
    if ext == '.dat':
        twix = mapvbvd.mapVBVD(ksp_file)
        twix.image.flagRemoveOS = False
        ksp = twix.image['']
        ksp = np.squeeze(ksp)
    elif ext == ".npy":
        ksp = np.load(ksp_file)
    else: # not recognized type of file
        raise ValueError(f"Unsupported kspace file type: {ksp_file}")
        
    # resize 
    diff = ksp.shape[0] - Npoints
    if diff > 11:
        ksp = ksp[11:, :, :] #always (starting point is not always the same (Heuristic))
        ksp = ksp[:Npoints, :, :]
    else:
        ksp = ksp[diff:, :, :]
    ksp = ksp[:Npoints, :, :]
    
    # select current echo
    startIdx = borders[echo]
    endIdx = borders[echo + 1]
    
    ksp = ksp[startIdx : endIdx, : , :]
    
    if R != None:
        ksp = ksp[:, :, ::R]
    '''
    # clipping ksp
    ksp= ksp[:thresh_i+1:subsample,:,::subsample_shots]
    '''
    del twix
    ksp = np.moveaxis(ksp, [1,2], [2,1])
    ksp = np.reshape(ksp, (1, -1, 1, Ncoils), order = 'F') # fix this
    #ksp = np.reshape(ksp, (32,-1), order = 'F') # fix this
    ksp = np.squeeze(ksp).T
    # select certain coils 
    #ksp = ksp[8:8+12, :]
    return ksp

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def save_as_nifti(data, filename):
    # Create a NIfTI image object
    nifti_image = nib.Nifti1Image(data, np.eye(4))  # Assuming identity affine for simplicity

    # Save the NIfTI image to a file
    nib.save(nifti_image, filename)
    return

def read_nifti(filename):
    # Load the NIfTI image from the file
    nifti_image = nib.load(filename)
    
    # Get the data from the NIfTI image
    data = nifti_image.get_fdata()
    return data

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


def kspace_precond(mps, coord=None, psf = 0,
                   lamda=0, device=sp.Device(0), savename='',
                   oversamp=1.25):
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
    xp = np

    mps_shape = list(mps.shape)
    img_shape = mps_shape[1:]
    
    # don't oversample if PSF case where it is already oversampled
    if psf:
        img2_shape = img_shape
    else:
        img2_shape = [i * 2 for i in img_shape]
        
    ndim = len(img_shape)

    scale = sp.prod(img2_shape)**1.5 / sp.prod(img_shape)
    
    coordLen = coord.shape[:-1]
    
    with device:
        if psf == 0:
            coord2 = coord
        else: # don't pad if calculating PSF, bc it is already padded
            coord2 = coord * 2 # doesn't add new memory
            
        # make memmap 
        ones = xp.ones(coordLen, dtype=dtype)
        ones = np.reshape(ones,[1,-1])
        #ones = makeMemMap(ones, "ones_memmap")
        
        # calculate PSF
        psf = nufft_adjoint(ones, coord2, oshape=img2_shape ,oversamp=oversamp) # adds memory
        
        # free memory
        del ones

        # store in memmap
        p_inv_memmap = np.memmap(savename+"temp/p_inv_memap", dtype=np.csingle, mode='w+', shape=(mps_shape[0], coordLen[0]))
        
        count = 0
        for mps_i in mps: # choose current sensitivity map [Si]
            #print(f'\rmap: {count}/{mps_shape[0]}', end='')
            xcorr_fourier = 0
            
            # calculate the correlation between all the sansitivity maps by multiplying them in the frequency domain
            mps_i = sp.to_device(mps_i, device) # on GPU
            
            for mps_j in mps: # Xcorr = sum_i,j( | fft( Si * Sj^H ) | )
                mps_j = sp.to_device(mps_j, device)
                
                xcorr_fourier += cp.abs(sp.fft(mps_i * cp.conj(mps_j), img2_shape))**2
            
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

        p_inv_memmap = np.memmap(savename+"temp/p_inv_memap", dtype=np.csingle, mode='r+', shape=(mps_shape[0], coordLen[0]))
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
    #coord_GPU = sp.to_device(coord.copy(), device) # make a copy of coord on GPU
    
    coord_scaled = _scale_coord(coord.copy(), input_shape, oversamp)
    coord_scaled = sp.to_device(coord_scaled, device)
    
    output = interp.interpolate(output, coord_scaled, kernel='kaiser_bessel', width=width, param=beta)
    
    del coord_scaled
    output = sp.to_device(output) # move output to CPU
    
    output /= width**ndim

    return output 

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
    
    coord =  sp.to_device(coord) # to CPU
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

def forwardSeperateCoils(x, mps, coord, forwardOutShape, C, device=sp.Device(0)):
    # vector to collect output
    #output = np.empty(forwardOutShape, dtype=x.dtype)
    output = np.memmap("temp/output_memap", dtype=x.dtype, mode='w+', shape=forwardOutShape)
    
    memLeft = get_gpu_memory()[0] - cp.get_default_memory_pool().used_bytes()
    # figure out max that can fit 
    c_batch = 2
    c_batch = min(mps.shape[0], c_batch)
    
    steps = ceil(C / c_batch)
    
    for c in range(steps):
        #print(f'\r Forward: {c}/{forwardOutShape[0]}')
        # apply sensitiviy map operator 
        
        mps_c = mps[ c*c_batch : c*c_batch + c_batch, : , :, :] # select only one sensitivty map at a time 
        with device:
            mps_c = sp.to_device(mps_c, device) # on GPU [1, 160, 160, 160]
            #v = sp.to_device(x, device) # move x to device
            v = sp.to_device(np.repeat(np.expand_dims(x,axis=0), c_batch, axis=0) , device) # make c_batch copies and move to device
            
            v *= mps_c # apply linear operator [2, 160, 160, 160] * [2, 160, 160, 160] -->  [2, 160, 160, 160]
        # free memory
        del mps_c 
        
        # apply NUFFT 
        with device:
            #coord = sp.to_device(coord, device) # on GPU [M, 3] 
            # x is image, and using coordinates, you get the YB's frequency guess (can also use BART's NUFFT)
            v = nufft(v, coord, oversamp=1.25, width=4, device=device) # nufft [1, 160, 160, 160] -->  [1, M]

        output[c*c_batch : c*c_batch + c_batch,:] = v
        output.flush()
    output = np.memmap("temp/output_memap", dtype=x.dtype, mode='r+', shape=forwardOutShape)
    return output

def backwardSeperateCoils(v, mps, coord, N, C, device=sp.Device(0)):
    '''
    v -> [C, M] k-space domain information 
    N -> [N, N, N]
    '''
    memLeft = get_gpu_memory()[0] - cp.get_default_memory_pool().used_bytes()
    # figure out max that can fit 
    c_batch = 2
    c_batch = min(mps.shape[0], c_batch)
    
    steps = ceil(C / c_batch)
    
    output = 0 # initialized to scalar, but will become [N,N,N] vector
    for c in range(steps):
        v_c = v[c*c_batch : c*c_batch + c_batch,:] # only one coil's info [1, M]
        #v_c = np.reshape(v_c, [c_batch,-1])
        
        # apply nufft adjoint 
        with device:
            v_c = sp.to_device(v_c, device) # move to GPU
            #coord = sp.to_device(coord, device)  # move to GPU
            # keep coord on CPU as nufft will make a copy to the GPU 
            oshape = [c_batch]
            oshape.extend(N)
            
            v_c = nufft_adjoint(v_c, coord, oshape, oversamp=1.25, width=4, device=device) # [1, M] ---> [1, N, N, N]
            
        # multiply by mps ^ H
        mps_c = mps[ c*c_batch : c*c_batch + c_batch, : , :, :] # select only one sensitivty map at a time 
        with device:
            mps_c = sp.to_device(mps_c, device) # on GPU [1, 160, 160, 160]
            xp = device.xp
            
            # take complex conjugate of it 
            mps_c = xp.conj(mps_c)
            
            v_c = sp.to_device(v_c, device) # move to GPU
            
            v_c *= mps_c # apply linear operator [1, 160, 160, 160] * [1, 160, 160, 160] -->  [1, 160, 160, 160]
        del mps_c
        v_c = sp.to_device(v_c) # move to CPU 
        
        # reshape  [1, 160, 160, 160] -->  [160, 160, 160] aka to N
        output += np.sum(v_c, axis = 0)
    return output
    
def multPrecond(v, precond, device=sp.Device(0)):
    v = sp.to_device(v, device) # move to GPU
    precond = sp.to_device(precond, device)
    
    v = v *  precond # apply linear operator [8, M] * [8, M] -->  [8, M]
    
    v = sp.to_device(v) # move back to cpu
    precond = sp.to_device(precond)
    return v

def findMaxEigen(mps, coord, precond, dtype, forwardOutShape, device=sp.Device(0), eigen_iter = 30):
    N = mps.shape[1:]
    C = mps.shape[0] # apply coilSens operator one map at a time 
    
    # make inital x image 
    x = util.randn(N, dtype=dtype, device=sp.cpu_device) # on CPU [160, 160, 160]
    
    print(f'\rProgress: 0/{eigen_iter}', end='')
    
    eigen_all =[]
    
    for i in range(eigen_iter):
        # apply foward (move to k-space domain)
        v = forwardSeperateCoils(x, mps, coord, forwardOutShape, C, device=device) # -->  [8, M] ADDS MEMORY
        #v2 = forward(x, mps, coord, device)
        
        # multiply with preconditioner
        #v = multPrecond(v, precond, device=device) # [8, M] * [8, M] -->  [8, M] on GPU
        #v = v *  precond # elementwise multiplication on CPU
        v *= precond
        v.flush()
        
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
        
    return max_eig, eigen_all

def l1WaveletRecon(image_guess, wavelet_name=''):
    N = image_guess.shape
    w = wavelet.fwt(image_guess, wave_name=wavelet_name, axes=None, level=None)
    return w

def  l1WaveletRecon_H(k_guess, N, wavelet_name=''): # device of input doesn't matter, will be copied to CPU
    _, coeff = wavelet.get_wavelet_shape(N, wave_name=wavelet_name, axes=None, level=None)
    return wavelet.iwt(k_guess, N, coeff, wave_name=wavelet_name, axes=None, level=None)

def calcObjective(ksp, image_guess, mps, coord, lamda = 0.01, wavelet_name='', device=sp.Device(0)): # all should be in numpy arrays
    ksp_guess = forwardSeperateCoils(image_guess, mps, coord, ksp.shape, ksp.shape[0], device=device)
    r = ksp_guess - ksp # error 
    obj = ( 1 / 2 ) * ( np.linalg.norm(r).item()**2 )
    # plus add wavelet regularization to the objective function 
    obj += lamda * np.sum(np.abs(l1WaveletRecon(image_guess, wavelet_name=wavelet_name))).item()
    return obj

def updateDual(u, image_ext, mps, coord, ksp, precond, device):
    # fft forward (from image to kspace domain)
    ksp_guess = forwardSeperateCoils(image_ext, mps, coord, ksp.shape, ksp.shape[0], device=device) # outputs a memory map

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
    
    return u

def updatePrimal(image_guess, tau, u, mps, coord, N, C, device):
    '''
    performing the math operation to update the current image guess: image_guess = image_guess - tau * A^H (u)
    '''
    # apply backward model inverse FFT (back in image domain from kspace frequency domain)
    image_err = backwardSeperateCoils(u, mps, coord, N, C, device=device)
        
    image_guess += -tau * image_err
    del image_err
    
    return image_guess
    
def waveletRegularization(image_guess, wavelet_name, alpha, N, device):
    '''
    Performing regularization on the image by subtracting high frequency wavelet components
    L1 reg [ wavelet^H ( L1( lamda*tau, wavelet( image_guess ) ))] = [ wavelet^H ( L1_err_k-domain ))] = L1_err_image_domain
    '''
    W_img = l1WaveletRecon(image_guess, wavelet_name=wavelet_name) # in k-space now
    del image_guess
    
    # move to GPU
    W_img = sp.to_device(W_img, device)
    k_err = thresh.soft_thresh( alpha, W_img) # can do this on GPU with C kernel (or cpu)
    k_err = sp.to_device(k_err) # back to CPU
    del W_img
    
    # inverse wavelet transform 
    #image_err = l1WaveletRecon_H(k_err, kshape)
    image_guess = l1WaveletRecon_H(k_err, N, wavelet_name=wavelet_name) # wavelet only works on CPU
    
    return image_guess

def PD_alg(coord, ksp, precond, mps, max_eig, iterSave = [], max_iter = 100, savename='', wavelet_name='', lamda = None, device=sp.Device(0)):
    # image reconstruction with the partial dual algorithm 
    
    # make initial guesses
    C =  ksp.shape[0]
    kshape = ksp.shape
    N = mps.shape[1:]
    
    obj_all = [] # keep track of objective function 
    
    gamma_primal = 0 
    gamma_dual = 1
    
    tol = 0
    
    tau = 1 / max_eig
    
    #u = np.zeros(kshape, dtype=ksp.dtype) # [C, M] vector in k-domain
    
    # make mem map instead 
    u = np.memmap(savename+"temp/u_memap", dtype=ksp.dtype, mode='w+', shape=kshape)
    u[:] = 0 + 0j
    u.flush()
    
    image_guess = np.zeros(N, dtype=ksp.dtype) # [N, N, N] vector in image-domain
    
    sigma_min = np.min(np.abs(precond)).item() # absolute min value of the preconditioner 
    #resid = np.infty # we will try to make this into a small number, so initalize to big
    
    # initialize part that goes into the dual algorithm
    image_ext = image_guess
    
    print(f'\rProgress: {0}/{max_iter}', end='')
    
    # start primal-dual 
    for i in range(max_iter):
        
        # update dual 
        u = updateDual(u, image_ext, mps, coord, ksp, precond, device)
            
        # update primal ( image_guess = image_guess - tau * A^H (u) )
        image_old = image_guess.copy() # keep track of old image 
        
        image_guess = updatePrimal(image_guess, tau, u, mps, coord, N, C, device)
        
        # regularization term 
        # this part smooths the image by removing high wavelet frequency image information
        # therefore, high enough lamba (perameter that controls how much is removed) should be high enough the remove speckled image artifacts
        # but not too high to create smoothed image sans details
        # L1 reg [ wavelet^H ( L1( lamda*tau, wavelet( image_guess ) ))] = [ wavelet^H ( L1_err_k-domain ))] = L1_err_image_domain
        image_guess = waveletRegularization(image_guess, wavelet_name, lamda*tau, N, device)
        
        # update all terms 
        theta = 1 / (1 + 2 * gamma_dual * sigma_min)**0.5
        precond *= theta
        sigma_min *= theta
        tau /= theta
        
        # Extrapolate primal
        image_diff = image_guess - image_old
        #resid = np.linalg.norm(image_diff / tau**0.5).item()
        
        image_ext = image_guess + theta * image_diff 
        
        if (i%10 == 0 and i != 0) or i+1 == max_iter:  # only calculate objective every 10th (to speed up), skip the first iteration bc it is very large since the guess is random
            obj_all.append(calcObjective(ksp, image_guess, mps, coord, wavelet_name=wavelet_name))
            
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
            titleName = 'L1 wavelet PDHG MC precond, iter = ' + str(i+1)
            plt.suptitle(titleName)
            plt.tight_layout()
            plt.savefig(savename + "_" + str(i+1) + "iter.svg", format = 'svg')
            plt.show()
        
    return image_guess, obj_all

# function summarizing what's happening 
def performRecon(ksp, coord , iterSave=[], max_iter=50, saveName='', N = None, eigen_iter = 20, psf = 0, wavelet_name='db4', eg_val=None, lamda=0.01, device=sp.Device(0)): # all inputs should be on CPU
    '''
    ksp = k-space measurements [8, M]
    coord = k-space coordinattes [M, 3]
    mps = sensitivity maps [C, N, N, N]
    precond = previously computed preconditioning array [8, N]
    '''
    # make temp directory for memory maps , if it doesn't already exist
    try:
        os.makedirs('temp')
    except:
        pass
    
    if ksp.ndim == 1:
        ksp = np.reshape(ksp, [ksp.shape[0],1]).T
    elif ksp.shape[0] > ksp.shape[1]:
        ksp = ksp.T
        
    if  coord.shape[0] < coord.shape[1]:
         coord = coord.T
         
    print(cp.get_default_memory_pool().used_bytes())
    
    print('calculating maps...', end = ' ')
    mps = JsenseRecon(ksp, coord=coord, img_shape = N, device=sp.Device(0)).run()
    print('done!')
    
    # make mps a memeroy map 
    mps = makeMemMap(mps, filename = saveName+"temp/mps_memmap")

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()
    print(cp.get_default_memory_pool().used_bytes())
    gc.collect()
    
    #mps = sp.to_device(mps, sp.Device(0))
    #coord = sp.to_device(coord, sp.Device(0)) # move to GPU for this function 
    print('calculating preconditioning matrix...', end = ' ')
    precond = kspace_precond(mps, coord=coord, savename=saveName, device=sp.Device(0), psf = psf)
    print('done!')
    
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    print(cp.get_default_memory_pool().used_bytes())
    
    # step 1: calculate max eigen values of operator A.H * S * A, where A is the sense forward operator (NUFFT * coilSens) and S is the preconditioning matrix
    if eg_val != None:
        max_eig = eg_val
        e_all = [eg_val]
    else:
        print("calculating eigen value...")
        max_eig, e_all = findMaxEigen(mps, coord, precond, ksp.dtype,  ksp.shape, eigen_iter=eigen_iter , device=device)
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

    
    print("starting image recon with Partial-Dual algorithm...")
    # step 2: start primal dual algorithm
    image, obj_values = PD_alg(coord, ksp, precond, mps, max_eig, max_iter=max_iter, iterSave=iterSave, savename=saveName, wavelet_name=wavelet_name, lamda=lamda, device=device)
    print("done!")
    
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()
    print(cp.get_default_memory_pool().used_bytes())
   
    # make range for epochs showing algroithm error
    obj_range = list(range(0, max_iter, int(max_iter / len(obj_values))))[1:]
    if obj_range[-1] != max_iter:
        obj_range.append(max_iter)
    
    plt.plot(obj_range, obj_values)
    plt.title("Partial-Dual Image Recon Alg Error")
    plt.xlabel("epoch")
    plt.ylabel("Objective Function")
    plt.show()
    
    # delete memory map
    if os.path.exists("temp/p_inv_memap"):
        os.remove("temp/p_inv_memap")
    if os.path.exists("temp/output_memap"):
        os.remove("temp/output_memap")
    if os.path.exists("temp/mps_memmap"):
        os.remove("temp/mps_memmap")
    # delete memory map
    if os.path.exists("temp/u_memap"):
        os.remove("temp/u_memap")
           
    return image, obj_values # return image and objective func scores 

if __name__ == "__main__":
    FOV = 0.24
    
    kloc_name = '/export02/data/Jana/NewSpiral/YB_1Echo_240fov_6msRO_0p8mm_1alpha/233_Jan25_yarn_Jana/traj/YB_1Echo_240fov_6msRO_0p8mm_1alpha_allInterleaves_nyTest_2.npy'
    ksp_file = '/export02/data/Jana/NewSpiral/YB_1Echo_240fov_6msRO_0p8mm_1alpha/233_Jan25_yarn_Jana/twix/meas_MID00071_FID19969_JS_yarn_matching.dat'
    
    coord = np.load(kloc_name) 
    npts, _, Nshots = coord.shape
    del coord 
    
    # make memroymap
    coord = np.memmap("coord_recon", dtype=np.float32, mode='r+', shape=(npts*Nshots, 3))
    ksp = np.memmap("ksp_recon", dtype=np.csingle, mode='r+', shape= (32, npts*Nshots))
    
    
    savename = ksp_file.split('.')[0] + "_imgPDHGL1Wavelet_MCprecon"
    
    res = 0.8 #mm
    N = [int(FOV*1000/res)]*3
    pdhg_mc_img, obj_values = performRecon(ksp, coord, max_iter=200, iterSave=[5, 10, 20, 30, 50, 75, 90, 100, 125, 150, 175, 190], savename=savename, continuation = 32, N=N,  device=sp.Device(0))
    
    
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
