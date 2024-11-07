#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 15:34:11 2023

@author: janastri

# add rotations based on missing points 
"""

import numpy as np
import math

def manualRot(u, v):
    # rotation to go from u to v
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)

    # vector normal to both
    n = np.cross(u,v)
    n = n / np.linalg.norm(n)
    
    # vector normal to the n and u
    t = np.cross(n,u)
    
    # angle along great circle from u to v
    alpha = math.atan2(np.dot(v,t), np.dot(v,u))
    
    T = np.array([u,t,n]).T
    
    R = np.array([[math.cos(alpha), - math.sin(alpha), 0],[math.sin(alpha), math.cos(alpha), 0],[0, 0, 1]])
    
    # checking 
    # should be that:
    # T @ R @ T.T @ u == v
    '''
    from mayavi import mlab
    mlab.figure()
    # og point 
    mlab.points3d(u[0], u[1], u[2], scale_factor = 1, color = (1,0,0))
    # desired 
    mlab.points3d(v[0], v[1], v[2], scale_factor = 0.4, color = (1,0,1))
    # new
    mlab.points3d((T@R@T.T@u)[0], (T@R@T.T@u)[1], (T@R@T.T@u)[2], scale_factor = 0.5, color = (0,1,1))
    '''
    return T@R@T.T

def addRotationsHere(curentEchoPoints, AlltrajPoints, MissingPoints):
    # selecting points on trajectory (change for multiecho case to only select first ech0)
    
    # radius of traj
    r_traj = np.linalg.norm(curentEchoPoints,axis = 1)
    
    manuallyAddedShots = []
    
    Rall = []
    for point in MissingPoints:
        # midpoint between them
        
        # radius of this new point
        r_point = np.linalg.norm(point)
        
        # closest radius value on traj to the point's traject
        idx = np.argmin(abs(r_traj - r_point))
        
        trajPoint = curentEchoPoints[idx]
        desiredPoint = point * r_traj[idx]/r_point # radius is changed so that it fits the radius of the closest point
        
        # the two points should be at the same radius value:
        #np.linalg.norm(trajPoint) - np.linalg.norm(desiredPoint)
        
        # perform rotation
        R = manualRot(trajPoint, desiredPoint)
        Rall.append(R)
        
        # perform rotations 
        newShow = (R@AlltrajPoints.T).T
        
        manuallyAddedShots.append(newShow)
        '''
        # checking 
        from mayavi import mlab
        mlab.figure()
        # og traj
        mlab.points3d(trajPoints[:,0], trajPoints[:,1], trajPoints[:,2], scale_factor = 0.4)
        # traj point
        mlab.points3d(trajPoint[0], trajPoint[1], trajPoint[2], scale_factor = 1, color = (1,0,0))
        # desired point 
        mlab.points3d(desiredPoint[0], desiredPoint[1], desiredPoint[2], scale_factor = 1, color = (0,0,1))
        # new show 
        mlab.points3d(newShow[:,0], newShow[:,1], newShow[:,2], scale_factor = 0.4, color = (1,0,1))
        '''
    return manuallyAddedShots, Rall
    
