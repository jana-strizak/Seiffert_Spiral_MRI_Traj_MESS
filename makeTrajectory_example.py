#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:19:26 2024

@author: janastri

This script makes a new trajectory, finds all the nyquist sampled shots, and performs with T2* and without PSF analysis
"""

import math 
import numpy as np

from SeiffertSpiralMRITraj import makeTrajectroy, createShots
from SeiffertSpiralMRITraj.recon.PSFmaker import makePSF
from SeiffertSpiralMRITraj.recon.phantomSim import phantomSimImage, plotImage
from SeiffertSpiralMRITraj.recon.imageRecon import performRecon, save_as_nifti, readMRIdataMESS


def formatKLocations(kloc, FOV):
    # the coordinates in kspace must be properly formated for image reconstruction as shown bellow 
    # Input: a Npoints by 3 by Nshots numpy array 
    # output: a Npoints*Nshots by 3 array scaled by the FOV      
    kloc = np.reshape(np.transpose(kloc,[2,0,1]), [kloc.shape[0]*kloc.shape[2], kloc.shape[1]])
    kloc = kloc.T * FOV
    kloc = np.array(kloc, dtype = np.float32)
    return kloc

if __name__ == "__main__":
    
    #%matplotlib auto
    save = 1
    #Input: #echos, resolution, fov, R, 
    resolution = 2e-3 # m
    fov = 240e-3 # m (needs to be this for the brain)
    
    #fixed: max v (max grad), max accel (slew rate), readout duration (40ms T2* decay @7T).
    gamma = 267.5153151e6 #rad/sT
    gradMax = 30e-3 #T/m
    vkMax = gradMax *gamma/(2*math.pi) # sT
    slewMax = 180 #T/m/s
    samplingRate = 1e-6 # s (max)

    # radius modulation shape 
    nLeafs = 1
    readoutDur = 6e-3 # s readout for all echos 
    alpha = 0.5

    modulationType ="centerOut"
    #modulationType ="edgeToEdge"
    
    show = 1
    
    myTraj = makeTrajectroy(nLeafs, resolution, fov, readoutDur, gamma, gradMax, vkMax, slewMax, samplingRate, alpha, modulationType ="centerOut", save = 1, show = 1)
    
    # create fully sampled trajectroy 
    myTraj_R1 = createShots(myTraj, R = 1, save = True)
    
    # create trajectroy that has is 6 times undersampled on the cartesian grid (nyquist test will be solved for sqrt(R) in all directions)
    myTraj_R6 = createShots(myTraj, R = 6, save = True)
    
    # create trajectroy that has is 6 times accelerated compared to the R = 1 case (fib rotations will be performed on 1/Ath of the points of the R = 1 fully sampled case)
    myTraj_A6 = createShots(myTraj, A = 6, save = True)
    
    # make Point Spread Function (PSF)
    makePSF(myTraj_R1, save = True)
    
    # PSF with T2* dephasing
    makePSF(myTraj_R1, T2_star = True, decayTime = 10, save = True)
    
    # make Point Spread Function (PSF)
    makePSF(myTraj_R6)
    
    
    # make Point Spread Function (PSF)
    makePSF(myTraj_A6)
    
    # making phantom simulations 
    # increasing number of reconstruction itterations could lead to a better image, but 50 should be enough for most cases
    simImage, groundTruth = phantomSimImage(myTraj_A6, reconIter = 50)

    
    # reconstructing real MRI data aquired along a trajectroy
    # reconstructing a single- and multi-echo trajectory is done the same way. For the multi-echo case, each image will be reconstructed one at a time 
    twixLocation = "./exampleData/meas_MID00031_FID44590_JS_yarn_UP_singleEcho_ro6_a2_out_R6.dat"
    
    # importing corresponding trajectroy
    trajLocation = "./exampleData/YB_1Echos_240FOV_6msRO_0p8mmRes_2alpha_SpiralOut/A6/YB_1Echos_240FOV_6msRO_0p8mmRes_2alpha_SpiralOut_A6_traj.pickle"    # load traj
    import pickle
    with open(trajLocation, 'rb') as f:
        myTraj_A6 = pickle.load(f)
      
    # extract kspace from twix file
    kspace = readMRIdataMESS(twixLocation, myTraj_A6)
    
    #kspace2 = readMRIdata(twixLocation, myTraj_A6.kspaceCoord.shape[0])
    
    # format kspace coordinate locations for recon
    kloc = formatKLocations( myTraj_A6.kspaceCoord, myTraj_A6.FOV)
    # get matrix size 
    N = myTraj_A6.N
    
    # compressed sensing value 
    lamda = 0.0001
    
    # reconstruct phantom image sampled along non-cartesian k-space coordinates
    image, error = performRecon(kspace, kloc.T, max_iter = 30, N = N, eigen_iter = 20, lamda=lamda)
    
    # plot reconstructed image
    plotImage(image, save = save, saveName = 'reconstructedMRIImage', plotTitle = "MRI Image Sampled Along Trajectroy")

    # save 
    save_as_nifti(abs(image), "image.nii")
    