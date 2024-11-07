#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 21:21:42 2023

@author: janastri
"""


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pickle

def makeCmap(traj): 
    colours = np.arange(traj.nLeafs)
    #pointsOneLeaf = round(traj.coordinates.shape[0]/traj.nLeafs)
    
    if traj.nLeafs > 1:
        pointsOneLeaf = traj.localmaxIdx[1:] - traj.localmaxIdx[:-1]
        pointsOneLeaf[-1] += 1
    else:
        pointsOneLeaf = [traj.coordinates.shape[0]]
        
    cMap = []
    
    for e in range(traj.nLeafs):
        cMap_curr = np.tile(colours[e],(pointsOneLeaf[e],1))
        cMap.append(cMap_curr)
        
    #cMap = np.tile(colours,(pointsOneLeaf,1))
    #cMap = np.reshape(cMap.T,cMap.shape[0]*cMap.shape[1])
    cMap = np.squeeze(np.vstack(cMap))
    return cMap

def plotBackgroundSphere(ax, res):
    # Create spherical coordinates
    theta = np.linspace(0, np.pi, 75)  # Polar angle
    phi = np.linspace(0, 2 * np.pi, 150)  # Azimuthal angle
    theta, phi = np.meshgrid(theta, phi)
    
    # Radius based on the resolution parameter
    r = 1 / (2 * res)  # Sphere radius
    
    # Convert spherical coordinates to Cartesian coordinates
    x1 = r * np.sin(theta) * np.cos(phi)
    y1 = r * np.sin(theta) * np.sin(phi)
    z1 = r * np.cos(theta)
    
    # Plotting the background sphere
    ax.plot_surface(x1, y1, z1, color='lightblue', alpha=0.2, edgecolor='none')
    return 

def plotSingleInterleave(traj, coord = None):
    if coord is None:
        coord = traj.coordinates
        
    # Create a new figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor((1, 1, 1))  # Set background color
    
    # Making Color map
    cMap = makeCmap(traj)
    
    # Plot the trajectory
    scatter = ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c=cMap, cmap='gist_rainbow', s=1)
    
    # Add color bar
    #plt.colorbar(scatter, ax=ax, label='Color map')

    # Customizing the orientation axes
    ax.set_xlabel('X Axis', color='black')
    ax.set_ylabel('Y Axis', color='black')
    ax.set_zlabel('Z Axis', color='black')

    # plotting Background sphere
    plotBackgroundSphere(ax, traj.res)
    plt.show()
    
    # plot each echo seperately as well if it is multi-echo
    if traj.nLeafs > 1:
        colormap = cm.get_cmap('gist_rainbow')
    
        for i in range(0, traj.nLeafs):
            # make new plot for this echo/leaf
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor((1, 1, 1))  # Set background color
            
            color = colormap(i/(traj.nLeafs-1))
            
            selected_points = coord[cMap == i]
            ax.scatter(selected_points[:, 0], selected_points[:, 1], selected_points[:, 2], color=color[:-1], s=1)
            
            # Customizing the orientation axes
            ax.set_xlabel('X Axis', color='black')
            ax.set_ylabel('Y Axis', color='black')
            ax.set_zlabel('Z Axis', color='black')
            
            # Highlighting the origin
            ax.scatter(0, 0, 0, color='red', s=2, label='Origin')
            ax.tight_layout
            plt.show()
    return 

def plotAllInterleaves(allInterleaves, traj, Nplot=None, echo=None, figure = True, tubeSize = 0.5):
    # specifying how many interleaves/shots to plot 
    Nplot =  Nplot if Nplot else allInterleaves.shape[2]
    
    # Create a new figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor((1, 1, 1))  # Set background color
    
    # if it's the single echo case or if one echo was selected for plotting
    if traj.nLeafs == 1 or not(echo == None):
        # select colormap
        #colormap = cm.get_cmap('gist_rainbow')
        colormap = cm.get_cmap('winter')
        
        traj.localmaxIdx[-1] = allInterleaves.shape[0]
        
        # determing start and end index along the trajectory to plot
        start_i = traj.localmaxIdx[echo] if not(echo == None) else 0
        end_i = traj.localmaxIdx[echo + 1] if not(echo == None) else allInterleaves.shape[0]
        
        # plot each shot seperately
        for i in range(Nplot):
            # determing color of points
            color = colormap(i/(Nplot))
            
            # select shot to be plotted
            currentShot = np.squeeze(allInterleaves[start_i:end_i,:,i])
            
            # plot this interleave/shot
            ax.scatter(currentShot[:, 0], currentShot[:, 1], currentShot[:, 2], color=color, s=1)
            
    else: # multi-echo case and no echo was specified
        # Making Color map for all echos
        cMap = makeCmap(traj)
        colormap = cm.get_cmap('gist_rainbow')
        
        for i in range(Nplot):
            # select shot to be plotted
            currentShot = np.squeeze(allInterleaves[:,:,i])
            
            # plot
            ax.scatter(currentShot[:, 0], currentShot[:, 1], currentShot[:, 2], c=cMap, cmap='gist_rainbow', s=1)
    # Customizing the orientation axes
    ax.set_xlabel('X Axis', color='black')
    ax.set_ylabel('Y Axis', color='black')
    ax.set_zlabel('Z Axis', color='black')
    
    # Highlighting the origin
    ax.scatter(0, 0, 0, color='black', s=2, label='Origin')

    plt.show()
    return

# plot points on outter surphase for each echo
def plotSurfaceSphere(allInterleaves, traj):
    # make colormap 
    colormap = cm.get_cmap('gist_rainbow')
    
    # radii to plot at, choose concentric spheres at 4 different radii to model the distribution of points in 3D along the radial direction
    plotR = [round(traj.kmax/8), round(traj.kmax/4), round(traj.kmax/2), traj.kmax]
    
    # number of echos
    Nechos = traj.nLeafs

    # itterating through images 
    for e in range(0, Nechos):
        # create colormap for each echo to differentiate them
        color = colormap(e/(Nechos))
        
        # selecting only one image/echo
        currentIm = allInterleaves[traj.localmaxIdx[e]:traj.localmaxIdx[e+1],:,:]
    
        # reshape t0 2D array for plotting
        currentIm2D = np.reshape(np.transpose(currentIm,[0,2,1]), [currentIm.shape[0]*currentIm.shape[2], currentIm.shape[1]])
        
        # calculate distance from the origin (radius) for each points
        r = np.linalg.norm(currentIm2D, axis=1)

        # 4 X 4 grid
        #fig = plt.figure(figsize=(12, 10))
        #axes = [fig.add_subplot(2, 2, i + 1, projection='3d') for i in range(4)]
        
        # 1 X 4 grid
        fig = plt.figure(figsize=(20, 5))  # Adjust the figure size as needed
        axes = [fig.add_subplot(1, 4, i + 1, projection='3d') for i in range(4)]

        #plot echo radius seperately
        for idx, i in enumerate(plotR):
            ax = axes[idx]
            
            # selecting only points inside a small shell around that radius 
            mask = (i - traj.dk_nyquist < r) & (r < i + traj.dk_nyquist)
            pointsPlot = currentIm2D[mask.nonzero()[0]]
            
            # now plot
            # Create a new figure
            ax.scatter(pointsPlot[:, 0], pointsPlot[:, 1], pointsPlot[:, 2], color=color, s=0.0005)
            ax.set_title("radius = %d"%i, fontsize = 16)
            # Customizing the orientation axe
            ax.set_xlabel('X Axis', color='black',fontsize = 16)
            ax.set_ylabel('Y Axis', color='black',fontsize = 16)
            ax.set_zlabel('Z Axis', color='black',fontsize = 16)
            
            ax.tick_params(axis='both', which='major', labelsize=16)  # Set major ticks
            ax.tick_params(axis='both', which='minor', labelsize=16)   # Set minor ticks (if applicable)

            # Highlighting the origin
            ax.scatter(0, 0, 0, color='black', s=2, label='Origin')
        fig.suptitle("Echo %d Radii Visualization"%e, fontsize=25)
        plt.tight_layout()
        plt.show()
    return 

'''
# only plot one echo at a time 
def plotSurfaceSphere2(allInterleaves, traj, e, save = 0, name = None):
    colormap = cm.get_cmap('gist_rainbow')
    
    # radii to plot at
    plotR = [round(traj.kmax/8), round(traj.kmax/4), round(traj.kmax/2), traj.kmax]
    
    # determing color of points
    color = colormap(e/(traj.localmaxIdx.shape[0]-2))
    
    # selecting only one image
    currentIm = allInterleaves[traj.localmaxIdx[e]:traj.localmaxIdx[e+1],:,:]
    
    # reshape t0 2D array
    currentIm2D = np.reshape(np.transpose(currentIm,[0,2,1]), [currentIm.shape[0]*currentIm.shape[2], currentIm.shape[1]])
    r = np.linalg.norm(currentIm2D, axis=1)
    
    #number of points in the image
    #Npoints = traj.localmaxIdx[1:] - traj.localmaxIdx[:-1]
    
    for i in plotR:
        # selecting only points inside a certain area 
        mask = (i - traj.dk_nyquist < r) & (r < i + traj.dk_nyquist)
        pointsPlot = currentIm2D[mask.nonzero()[0]]
        
        mlab.figure(bgcolor=(1,1,1), size=(1000, 1000))
        #print(i)
        #mlab.points3d(pointsPlot[:,0], pointsPlot[:,1], pointsPlot[:,2], scale_factor=i*0.01, color=tuple(color[:-1]))
        scale = (i/traj.kmax)**0.5
        mlab.points3d(pointsPlot[:,0], pointsPlot[:,1], pointsPlot[:,2], scale_factor=0.5*scale, color=tuple(color[:-1]))
        #mlab.title("radius = %d, image %d"%(i,e), color = (0,0,0))
        
        orientation_axes = mlab.orientation_axes()
        orientation_axes.axes.x_axis_caption_actor2d.caption_text_property.color = (0, 0, 0)
        orientation_axes.axes.y_axis_caption_actor2d.caption_text_property.color = (0, 0, 0)
        orientation_axes.axes.z_axis_caption_actor2d.caption_text_property.color = (0, 0, 0)
        
        if save:
            mlab.savefig(name + '_r' + str(i) + '.tiff')
            
    
    return
'''
#Gradient plotting 
def plotWaveform(t, coord, title='', ylabel=''):
    # Plotting the x,y,z 
    fig, axs = plt.subplots()
    
    if coord.shape[1] != 4:
        r = np.linalg.norm(coord,axis=1)
    else: 
        r = coord[:,3]
        
    # plot 
    axs.plot(t, coord[:,0], label = 'x')
    axs.plot(t, coord[:,1], label = 'y')
    axs.plot(t, coord[:,2], label = 'z')
    
    # don't plot r for slew
    if  title != 'Slew Rate':
        axs.plot(t, r, label = 'r')
        
    axs.legend(fontsize=14, loc='lower right')
    axs.set_title(title, color = (0,0,0), fontsize=18)
    axs.set_xlabel('Time (ms)', fontsize=18)
    axs.set_ylabel(ylabel, fontsize=18)
    fig.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.show()
    return

# testing 
if __name__ == '__main__':
    
    '''
    # import test trajectory 
    from pickle import load
    with open('myTestTraj.pickle', 'rb') as f:
        myTraj = load(f)
    
    # plot single interleave 
    plotSingleInterleave(myTraj)
    
    # make other interleaves 
    from FibanocciLatticeRotation_MultiProcessing import FibanocciRotate
    Nfib = 100 # few rotations
    allInterleaves = FibanocciRotate(myTraj.coordinates, Nfib)
    
    # plot all interleaves 
    plotAllInterleaves(allInterleaves, myTraj, Nplot=2)
    '''
    
    with open("allInterleaves_TEST.pickle", "rb") as f:
        allInterleaves = pickle.load(f)
    with open("myTraj_TEST.pickle", "rb") as f:
        myTraj = pickle.load(f)
        
    #plotSingleInterleave(myTraj)
    #allInterleaves = FibanocciRotate(myTraj.coordinates, Nfib)
    
    plotSurfaceSphere(allInterleaves, myTraj)