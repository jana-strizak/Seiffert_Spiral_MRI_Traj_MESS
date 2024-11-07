#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 11:41:35 2023

@author: janastri

modified so the density plots are number of cartesian points in column if it was the same length as the sphere 

"""

# Plotting density of points for many rotations 

import pickle 
import numpy as np
import matplotlib.pyplot as plt
from math import floor
#from matplotlib_inline.backend_inline import set_matplotlib_formats
#set_matplotlib_formats('pdf', 'svg') 

def plotHistxyz(myTraj, checkPoints_allInter, imageIndx=0, heatmap0_prev = None, heatmap1_prev = None, max_value = None, heatmap3_prev = None, top_color_bounds = 0.4, title = None):
    # Reshaping 
    pointsAll = np.reshape(np.transpose(checkPoints_allInter,[2,0,1]), [checkPoints_allInter.shape[0]*checkPoints_allInter.shape[2], checkPoints_allInter.shape[1]])
    
    # Radius of points
    rPoints = np.linalg.norm(pointsAll,axis=1)
    
    # x , x, z
    label1 = ['x','x','z']
    axis1 = [pointsAll[:,0], pointsAll[:,0], pointsAll[:,2]]
    # y , z, y
    label2 = ['y', 'z', 'y']
    axis2 = [pointsAll[:,1], pointsAll[:,2], pointsAll[:,1]]
    
    # bins same for all plots
    xedges = np.linspace(-myTraj.kmax, myTraj.kmax, num = floor(myTraj.kmax/myTraj.dk_nyquist)+1)
    yedges = np.linspace(-myTraj.kmax, myTraj.kmax, num = floor(myTraj.kmax/myTraj.dk_nyquist)+1)
    
    # radius 
    xMid = (xedges[1:] + xedges[:-1])/2
    yMid = (yedges[1:] + yedges[:-1])/2
    xGrid, yGrid = np.meshgrid(xMid,yMid)
    d = np.sqrt(xGrid**2 + yGrid**2)
    dInside = (d < np.max(rPoints))*d
    a = np.sqrt(np.max(rPoints)**2 - dInside**2)
    aPlot = a.T
    
    numPointsColumn = np.floor(aPlot/myTraj.dk_nyquist) + 1
    
    # making 3 plot figures
    fig0, axs0 = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    
    for i in range(0,len(label1)):
        # histogram 
        H, xedges, yedges = np.histogram2d(axis1[i], axis2[i], bins=(xedges, yedges)) 
        Hplot = H.T

        #cartesian normalized no center circle
        H_normalized = Hplot/numPointsColumn
        heatmap0 = axs0[i].imshow( H_normalized, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='hot')
        
        #axs0[i].set_title('Radius Normalized')
        axs0[i].set_xlabel(label1[i], fontsize = 22)
        axs0[i].set_ylabel(label2[i], fontsize = 22)
        axs0[i].tick_params(axis='both', which='major', labelsize=19)
        
        
        #cartesian normalized with center circle
        centerZeroedOut = (d > 0.1*max(rPoints)) * Hplot/numPointsColumn
     
        if max_value == None: 
            max_value = int(np.max(centerZeroedOut) * top_color_bounds)
       
        heatmap0.set_clim(0, max_value)
        
    # keeping track of other echo's data to make sure the colormap is the same
    if heatmap0_prev == None: 
        heatmap0_prev = heatmap0
  
    # Create a common colorbar for all heatmaps
    cbar = fig0.colorbar(heatmap0_prev, ax=axs0, shrink=0.6)
    cbar.ax.tick_params(labelsize=18)
    
    # Set a common title for the figure
    fig0.suptitle( title, fontsize = 20)
    plt.show()

    return heatmap0_prev, heatmap1_prev, max_value, heatmap3_prev

def plotHistEachImage(allInterleaves, myTraj, top_color_bounds = 0.4, heatmap0_prev = None, max_value = None, echosPlot = None, title = "Cartesian Normalized Density Plots"):
    pointsOneLeaf = int(myTraj.coordinates.shape[0]/myTraj.nLeafs)
    
    heatmap1_prev = None
    heatmap3_prev = None
    
    if echosPlot == None:
        echosPlot = myTraj.nLeafs
    
    for i in range(0, echosPlot):
        # current image
        allInterleaves_leaf = allInterleaves[i*pointsOneLeaf:(i+1)*pointsOneLeaf,:,:]
        
        if echosPlot > 1:
            title = title + ", Image = " + str(i)
            
        # plot histogram in 3 directions
        heatmap0_prev, heatmap1_prev, max_value, heatmap3_prev = plotHistxyz(myTraj, allInterleaves_leaf, i, heatmap0_prev = heatmap0_prev, heatmap1_prev = heatmap1_prev, max_value = max_value, heatmap3_prev = heatmap3_prev, top_color_bounds = top_color_bounds, title = title)

    return heatmap0_prev, max_value

def plotDiffRadii(checkPoints_allInter):
    
    return 

if __name__ == "__main__":
    '''
    with open('myTestTraj_2Echos.pickle', 'rb') as f:
        myTraj = pickle.load(f)
    '''
    with open("allInterleaves_TEST.pickle", "rb") as f:
        allInterleaves = pickle.load(f)
    with open("myTraj_TEST.pickle", "rb") as f:
        myTraj = pickle.load(f)
        

    plotHistEachImage(allInterleaves, myTraj)
    