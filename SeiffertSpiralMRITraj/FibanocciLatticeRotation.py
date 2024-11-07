#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 00:09:43 2023

@author: janastri
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#includes
import math
import numpy as np
import multiprocessing as mp
import pickle 

# define kernel
def rodrigues_formula(k, cos_a, sin_a, aor):
    return k*cos_a + np.cross(aor.T, k)*sin_a + np.dot(aor, k)*(1-cos_a)*aor


#Method Definitions
def makeRotMat(axisVec, angle):
    '''
    axisVec: (1x3 array)
    angles: 1D array of angles in radians
    '''
    # normalize axisVec
    axisOfRot = np.nan_to_num(axisVec/np.linalg.norm(axisVec))

    # initialize output array
    R = np.eye(3)

    # calculate rotation matrix for current angle using kernel
    R = np.apply_along_axis(rodrigues_formula, axis=1, arr=R, cos_a=np.cos(angle), sin_a=np.sin(angle), aor=axisOfRot)
    return R.T

def applyTransformations(R1, R2, axisAngles, v1, oneInterleaveArray, g):
    N = axisAngles.shape[0]
    
    allInterleaves = np.zeros((oneInterleaveArray.shape[0],oneInterleaveArray.shape[1], N))
    
    for j, (R1,R2) in enumerate(zip(R1, R2)):

        # kspace rotation
        oneInterleaveArrayNew = (R2@R1@oneInterleaveArray.T).T
        
        #axis rotation
        vAxis = (R2@R1@v1.T).T
        
        angleAxis = axisAngles[j]
        Raxis = makeRotMat(vAxis, angleAxis)
        oneInterleaveArrayAxisRotated = (Raxis@oneInterleaveArrayNew.T).T
        allInterleaves[:,:,j] = oneInterleaveArrayAxisRotated #adds all points to 3rd direction
        
        # roate to create all gradients
        g_allInter = None
        
        if g:
            g_allInter = np.zeros((g.shape[0],g.shape[1],N))
            gInterleave = (R2 @ R1 @ g.T).T
            gInterleave_AxisRotated = (Raxis @ gInterleave.T).T
            g_allInter[:,:,j] = gInterleave_AxisRotated
            
    return allInterleaves, g_allInter

def FibanocciRotate(oneInterleaveArray, res, N, v1 = None, axisAngles = None, g = None, mp_context = mp.get_context('forkserver')):
    '''
    Inputs:
        oneInterleaveArray: a numpy array with 3 columns representing the cartesian x,y,z locations, and rows being various points to be rotated
        N: number of endpoints along the Fibanocci Lattice (Also the number of rotations)
        ind: The index of the point in pointArray along which the rotation axis will be defined. Should be point on the sphere. Default is the first point.
    Outputs:
        a 3D numpy array with each slice representing points in 1 Fibanocci Rotation. First dimension is points, Second dimension is x,y,z locations, and third dimension is all Fibanocci Rotations
    '''
    goldenRatio = (1 + math.sqrt(5))/2 
    i = np.arange(N)
    
    theta = 2 *math.pi * i / goldenRatio
    phi = np.arccos(1 - 2*(i+0.5)/N)
    
    if type(axisAngles) != np.ndarray:
    #anxis angles to randomly rotated at
        axisAngles = np.random.uniform(0, 2*np.pi, N)
    #axisAngles = np.array([3.653307, 1.550821])
    
    if type(v1) != np.ndarray:
        try:
            v1 = oneInterleaveArray[-1,:] #first point of input array. The rotation axis will be defined using this point
        except:
            oneInterleaveArray = np.expand_dims(oneInterleaveArray,axis=0)
            v1 = oneInterleaveArray[-1,:]

    v2 = np.array([0,0, 1/res/2])
    
    n = np.cross(v1,v2) #a vector perpendicular to both
    #p = np.cross(n,v1) #another perpendicular vector 
    
    #make argument list 
    args_list1 = [(n, angle) for angle in phi]
    args_list2 = [(v1, angle) for angle in theta]
    
    # create argument list for parallel processing
    pool = mp_context.Pool()
    # create argument list for parallel processing
    #pool = mp.Pool()
    
    R1 = pool.starmap(makeRotMat, args_list1) #first rotation matrix
    R2 = pool.starmap(makeRotMat, args_list2) #second rotation matrix
        
    #oneInterleaveArrayNew = oneInterleaveArray@ R1 @ R2
    
    # finished with pool 
    pool.close()
    pool.join()
    
    #reshape output 
    #R1_all = np.stack(R1_tuple, axis=-1)
    #R2_all = np.stack(R2_tuple, axis=-1)
    
    #index of point acting as axis of rotations (v1)
    index = np.where(np.all(v1 == oneInterleaveArray, axis=1))
    
    g_allInter = None 
    
    # apply matrix multiplication 
    allInterleaves, g_allInter = applyTransformations(R1, R2, axisAngles, v1, oneInterleaveArray, g)
    
    return allInterleaves, axisAngles, g_allInter
