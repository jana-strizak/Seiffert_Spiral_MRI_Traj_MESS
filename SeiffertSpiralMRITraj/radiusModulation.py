#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This defines the radiusFunc, which is used to modulate the Sieferd Spiral from start to finish
Define function to perform on radius 
rules: 
y must go from -1 to 1 
x must start at zero
must include refelction about horizontal center of function when indicated
"""
import math
import numpy as np 
#from YarnballClass_condensed import Yarnball
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def scale(array, lowerBound, upperBound):
    normArray = (array - np.min(array))
    normArray = normArray/np.max(normArray)
    rescaledArray = normArray*(upperBound - lowerBound) + lowerBound
    return rescaledArray
    
def radiusFunc(startEndPoint, alpha, tsingle, Nechos, rMax):
    
    multiplier = 1
    
    if startEndPoint == "none":
        rmod = rMax * np.ones_like(tsingle)
        rmod = np.tile(rmod,Nechos)
        
    elif startEndPoint == "centerOut":
        sall = np.linspace(0, 1, num = tsingle.shape[0])**alpha
        
    elif startEndPoint ==  "edgeToEdge":
        
        # array to store radius modulation
        sall = np.zeros(tsingle.shape[0]*Nechos)
        
        for e in range(0,Nechos):
            L1 = round(tsingle.shape[0]/2)
            s1 =  multiplier*(np.linspace(1, 0, num = L1))**alpha
            
            # change sign of multiplier
            multiplier = -1* multiplier
            
            L2 = tsingle.shape[0] - L1
            s2 = multiplier*(np.linspace(0, 1, num = L2))**alpha
            
            s = np.hstack([s1, s2])
            sall[tsingle.shape[0]*e:tsingle.shape[0]*(e+1)] = s
            
        sall = scale(gaussian_filter(sall, sigma = 100), -1, 1)
        
        
    if startEndPoint != "none":
        rmod = rMax*(sall)
        #rmod = scale(gaussian_filter(rmod, sigma = 200), -rMax, rMax)
    
    return rmod
        
if __name__== "__main__":
    #Input: #echos, resolution, fov, R, 
    nLeafs = 7 #echos
    resolution = 1e-3 # m
    fov = 256e-3 # m
    R = 1 #full sampling
    readoutDur = 40e-3 # s @7T (diff at other field strengths)
    
    #fixed: max v (max grad), max accel (slew rate), readout duration (40ms T2* decay @7T).
    gamma = 267.5153151e6 #rad/sT
    gradMax = 80e-3 #T/m
    vkMax = gradMax *gamma/(2*math.pi) # sT
    slewMax = 200 #T/m/s
    samplingRate = 1e-6 # s (max)
    
    alpha = 0.5
    
    """
    /////////// Other User Inputs ////////////
    """
    showPlots = 1 # will display single multiple interleaves
    NinterleavesPlot = 5 # choose how many interleaves will be displayed, leave 'None' for all
    """
    //////////////////////////////////////////
    Part 1: Making the trajectory
    """
    """
    a) Define trajectory parameters
    """
    
    myTraj = Yarnball(samplingRate, readoutDur, nLeafs, vkMax, resolution, fov, modulationType ="edgeToEdge")
    myTraj.calcParameters()
    
    rmod = radiusFunc("edgeToEdge", alpha, myTraj.tLeaf, myTraj.nLeafs, myTraj.kmax)
    