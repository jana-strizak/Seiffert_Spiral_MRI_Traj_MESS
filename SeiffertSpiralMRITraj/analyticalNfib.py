#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 07:09:00 2023

@author: janastri
"""
import numpy as np
import math 
import pickle 
from SeiffertSpiralMRITraj.FibanocciLatticeRotation import FibanocciRotate

def findNfib(myTraj, show=0):
    
    # select first interleave 
    if myTraj.nLeafs == 1:
        coordinates = myTraj.coordinates
        
        points = [coordinates]
    else:
        #nPointsLeaf = round(myTraj.coordinates.shape[0]/myTraj.nLeafs)
        mid = 0 #round(myTraj.nLeafs/2) - 1
        #coordinates = myTraj.coordinates[:nPointsLeaf,:]
        #coordinates = myTraj.coordinates[nPointsLeaf*mid:nPointsLeaf*(mid+1),:]
        idx = 0
        startPointImage = myTraj.localmaxIdx[idx]
        endPointImage = myTraj.localmaxIdx[idx+1]
        coordinates = myTraj.coordinates[startPointImage:endPointImage,:]
        
        # seperate the points into 2 
        pointsLHS = coordinates[:round(np.shape(coordinates)[0]/2),:]
        pointsRHS = coordinates[round(np.shape(coordinates)[0]/2):,:]
        
        points = [pointsLHS, pointsRHS]
    
    Npoints = 0
    for p in points:
        # calculate radius 
        rpoints = np.linalg.norm(p, axis=1)
        mask = (myTraj.kmax - myTraj.dk_nyquist/2 < rpoints) & (rpoints <= myTraj.kmax + myTraj.dk_nyquist/2)
        
        # outter rim of points 
        outerRim = p[mask.nonzero()[0]]
        
        # distances 
        distances = np.linalg.norm(outerRim[1:,:] - outerRim[:-1,:], axis = 1)
        
        # cumulative sum 
        S = np.cumsum(distances)
        
        # number of points in rim 
        Npoints = Npoints + S[-1]/myTraj.dk_nyquist
    
    # number of radial trajectory points 
    Nradial = 4*np.pi*(myTraj.kmax/myTraj.dk_nyquist)**2 
    
    Nshots = math.ceil( Nradial / Npoints)
    
    return Nshots

if __name__ == "__main__":
    
    
    """
    Importing Test Trajectroy
    """
    '''
    with open('myTestTraj2_minTimeGradParam.pickle', 'rb') as f:
    #with open('myTestTraj_minTimeGradParam.pickle', 'rb') as f:
        myTraj = pickle.load(f)
    '''
    '''
    with open('myTestTraj_2Echos.pickle', 'rb') as f:
        myTraj = pickle.load(f)
    '''    
    with open("myTestTraj_2Echos_2mmRes.pickle","rb") as f:
        myTraj = pickle.load(f)
        
    
    NfibBest, allInterleaves = findNfib(myTraj)

    print("Nfib = ", NfibBest)