#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 15:47:43 2023
@author: janastri
Note: When plotting, make sure you have IPython 8.10.0 -- An enhanced Interactive Python in you environment and run '%gui qt' in the command line.

changed radius modulation function in yarnballClass
changed plots to be density plots
"""
import math
import numpy as np
from matplotlib_inline.backend_inline import set_matplotlib_formats

# script to make single interleave
from SeiffertSpiralMRITraj.trajectoryDesign import Yarnball
# script to find best m
from SeiffertSpiralMRITraj.mOptimization import findOptimalM
# script to perform density assessment 
from SeiffertSpiralMRITraj.densityPlots import plotHistEachImage # normalized by points in cartesian column with length equal to the sphere's cross section 
# script to excepute minTimeGradient.c script by S. Vaziri and M. Lustig “The Fastest Gradient Waveforms” from: http://people.eecs.berkeley.edu/~mlustig/Software.html
from SeiffertSpiralMRITraj.MRIGradientCalculation import createGradients
# script to create all interleaves by rotating by Fibanocci lattice (random axis rotations)
from SeiffertSpiralMRITraj.FibanocciLatticeRotation import FibanocciRotate
# Trajectory plotting functions 
from SeiffertSpiralMRITraj.plottingFunctions import plotSingleInterleave, plotAllInterleaves, plotSurfaceSphere
# performing nyquist check 
from SeiffertSpiralMRITraj.nyquistCheck import findNfibNyquist

import os
import pickle
import shutil

def clear_directory(directory):
    # Check if the directory exists
    if os.path.exists(directory):
        # Iterate over all files and subdirectories
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                # Remove the subdirectory and its contents
                shutil.rmtree(item_path)
            else:
                # Remove the file
                os.remove(item_path)
    else:
        print(f"The directory '{directory}' does not exist.")
    return

def giveName(nLeafs, fov, readoutDur, resolution, alpha, radiusModType):
    """
    Generates the name for the trajectory based on the specs of the trajectory's design
    Parameters
    ----------
    nLeafs : int
        The number of seperate segments of the trajectroy (Echoes)
    fov : float
        the field of view of the trajectroy, in meters. ex: 240e-3 m aka 240 mm
    readoutDur : float
        The desired readout duration of the trajectory in s. Note this is just a target and a more exact readout duration can be found traj.tTot.
    resolution : float
        Desired resolution in meters.
    alpha : float
        The exponent of the radial progression of the trajectory. If set to one the radius will evolve linearly (r), if set to 2, radial evaluestion will be faster (r^2), if set to 0.5, it will be slowers (r^0.5)
    radiusModType : string either "centerOut", or "edgeToEdge".
        Dictated how the trajectroy evolves in the radial direction including the start and end points in the alocated readout duration. 

    Returns
    -------
    name : string
        The trajectroy's name

    """
    name = 'YB_' + str(nLeafs) +"Echos_" + str(int(fov*1000)) + "FOV_" + str(int(readoutDur*1000))
    
    if int(((readoutDur* 1000)%1)*10) > 0:
        name = name + "p" + str(int(((readoutDur* 1000)%1)*10)) 
    
    name = name + "msRO_" + str(int(resolution*1000))
    
    if int(((resolution* 1000)%1)*10) > 0:
        name = name + "p" + str( int(((resolution* 1000)%1)*10))
    
    name = name + "mmRes_" + str(int(alpha))
    
    if int(alpha%1* 10) > 0:
        name = name + "p" + str(int(alpha%1* 10))
    
    name = name + "alpha"
    
    if radiusModType == "centerOut":
        name += "_SpiralOut"
    elif radiusModType == "edgeToEdge":
        name += "_SpiralInOut"
        
    return name 

def createShots(traj, N = None, R = None, A = None, save = False):
    """
    Calculates all the k- space shots for the sampling case
    Parameters
    ----------
    traj : object
        a single shot of your trajectroy which will be used to rotated and calculate the rest of the shots
    N : int, optional
        The number of desired shots without calculating any nyquist conditions.
        In this case, only fibanocci rotations with the desired number of shots will be performed, so no rotation matrices for extra shots is applicable.
        The default is None.
    R : int, optional
        The undersampling factor of the kspace, determined by the undersampling in all 3 directions ok kspace. 
        The nyquist criteria is solved for a cube root of R undersampling criteria so the total is R times undersampled.
        The default is None.
    A : int, optional
        The acceleration factor relative to the fully sampled R = 1 case. Fibanocci rotations will be created for 1/A of the shots required for the fully sampled case.
        In this case, only fibanocci rotations with the desired number of shots will be performed, so no rotation matrices for extra shots is applicable.
        The default is None.
    save : bool, optional
        Whether to save the data, including the trajectroy object and k-space locations for all shots, gradient values for 1 shot, rotations angles about the axis for each fibanocci rotated shot, and rotation matrices for the extra shots.
        The default is False.

    Returns
    -------
    kloc : object
        The trajectory object including the new 3D k-space location coordinates, number of shots, axis angles cooresponding to each shot's rotation about it's axis, and the rotation matrices for the extra shots, if applicable. 

    """
    # name trajectorty
    nameTraj = giveName(traj.nLeafs, traj.FOV, traj.durLeaf, traj.res, traj.alpha, traj.modulationType)
    
    # make new folder for this trajectroy (if it doesn't already exist) 
    current_directory = os.getcwd()
    nameDir = current_directory + '/' + nameTraj +'/'
    name = nameDir + nameTraj
    
    user_input = 'yes'
    
    # all  undersampled cases (solved by performing the nyquist test)
    if R != None:
        # make new folder for this R
        nameRDir = nameDir + 'R' + str(R)
        
        name = nameRDir + '/' + nameTraj + '_R' + str(R)

        # check if this trajectroy has already been calculated
        if os.path.isfile(name + '_traj.pickle'):
            print("Trajectory with this R value already exists..." + nameRDir)
            # Prompt the user to continue or exit
            user_input = input("Do you want to remake " + name + "? (yes/no): ").strip().lower()
        elif save:
            try:
                os.makedirs(nameRDir)
            except:
                pass
        
        if user_input == 'no' or user_input == 'No':
            print("Exiting the program.")
            # load trajectroy
            with open(name + '_traj.pickle', 'rb') as f:
                traj = pickle.load(f)
            return traj 
        
        elif user_input != 'yes' or user_input != 'Yes':
            # delete entire directory so files don't interfere 
            clear_directory(nameRDir)
            
            # make it again
            traj = findNfibNyquist(traj, R = R, save = save, name = name)
            print('done: ' + name)  
            return traj
        
    # choose exact num of shots
    elif N != None:
        nameRDir = nameDir + 'Nshots' + str(N)
        # check if this trajectroy has already been calculated
        if os.path.isdir(nameRDir):
            print("Trajectory with this R value already exists..." + nameRDir)
            # Prompt the user to continue or exit
            user_input = input("Do you want to remake " + name + "? (yes/no): ").strip().lower()
        elif save:
            os.makedirs(nameRDir)
            
        if user_input == 'no' or user_input == 'No':
            print("Exiting the program.")
            name = nameRDir + '/' + nameTraj + '_Nshots' + str(N)
            # load trajectroy
            with open(name + '_traj.pickle', 'rb') as f:
                traj = pickle.load(f)
            return traj 
            
        elif user_input != 'yes' or user_input != 'Yes':
            # delete entire directory so files don't interfere 
            clear_directory(nameRDir)
            
            name = nameRDir + '/' + nameTraj + '_Nshots' + str(N)
            if nLeafs == 1:
                v1 = traj.coordinates[-1,:]
            else:
                v1 = traj.coordinates[traj.localmaxIdx[1],:]
    
            # calculating all shots
            kloc, axisRotations, _ = FibanocciRotate(traj.coordinates, traj.res, N, v1 = v1)
            
            # saving rotation info
            traj.R = None 
            traj.Nshots = N
            traj.axisAngles = axisRotations
            traj.Rall = None
            traj.kspaceCoord = kloc
            traj.name = name 
            
            # saving
            if save:
                with open(name + '_traj.pickle', 'wb') as f:
                    pickle.dump(traj, f)
                np.save(name + "_allInterleaves_nyTest", kloc)
                np.savetxt(name + '_allInterleaves_nyTest' + '_axisAngles.dat', axisRotations,  delimiter=",")
                print('done: ' + name)
                return traj
        
    elif A != None:
        # choose undersampling factor relative to the R = 1 case
        if traj.Nshots_R1 is None:
            print("need to solve R = 1 case first to obtain total number of shots.")
            traj = findNfibNyquist(traj, R = 1, save = 0)
            print('done: R = 1 case requires ' + str(traj.Nshots) + ' shots.')  
            # save
            with open(name + '_traj.pickle', 'wb') as f:
                pickle.dump(traj, f)
        
        # make new fib distributed shots with acceleration factor
        nameRDir = nameDir + 'A' + str(A)
        name = nameRDir + '/' + nameTraj + '_A' + str(A)
        
        # check if this trajectroy has already been calculated
        if os.path.isfile(name + '_traj.pickle'):
            print("Trajectory with this R value already exists..." + nameRDir)
            # Prompt the user to continue or exit
            user_input = input("Do you want to remake " + name + "? (yes/no): ").strip().lower()
        elif save:
            os.makedirs(nameRDir)
            
        if user_input == 'no' or user_input == 'No':
            print("Exiting the program.")
            # load trajectroy
            with open(name + '_traj.pickle', 'rb') as f:
                traj = pickle.load(f)
            return traj 
        
        elif user_input != 'yes' or user_input != 'Yes':
            # delete entire directory so files don't interfere 
            clear_directory(nameRDir)
            
            # num of shots 
            Nfib = round(traj.Nshots_R1 / A)
            
            endPointEcho0 = max(traj.coordinates.shape[0]-1, traj.localmaxIdx[1])
            
            # calculating all shots
            kloc, axisRotations, _ = FibanocciRotate(traj.coordinates, traj.res, Nfib, v1 = traj.rotPoint)
            
            # saving rotation info
            traj.R = None 
            traj.A = A
            traj.Nshots = Nfib
            traj.axisAngles = axisRotations
            traj.Rall = None
            traj.kspaceCoord = kloc
            traj.name = name 
            
            # saving
            if save:
                with open(name + '_traj.pickle', 'wb') as f:
                    pickle.dump(traj, f)
                np.save(name + "_allInterleaves_nyTest", kloc)
                np.savetxt(name + '_allInterleaves_nyTest' + '_axisAngles.dat', axisRotations,  delimiter=",")
                
                print('done: ' + name)
                return traj
    return traj
    
def makeTrajectroy(nLeafs, resolution, fov, readoutDur, gamma, gradMax, vkMax, slewMax, samplingRate, alpha, modulationType ="centerOut", save = 0, show = 1):
    # name trajectorty
    if nLeafs > 1:
        modulationType = "edgeToEdge"
    nameTraj = giveName(nLeafs, fov, readoutDur/nLeafs, resolution, alpha, modulationType)
    
    # make new folder for this trajectroy (if it doesn't already exist) 
    current_directory = os.getcwd()
    nameDir = current_directory + '/' + nameTraj +'/'
    name = nameDir + nameTraj
    
    user_input = 'yes' # will be modifed if already exists
    
    # check if this already exists
    if os.path.exists(name + '_traj.pickle'):
        print("Trajectory" + name + '_traj.pickle' + "already exists...")
        # Prompt the user to continue or exit
        user_input = input("Do you want to remake " + name + "? (yes/no): ").strip().lower()
    elif save:
        os.makedirs(nameTraj)
       
    # make trajectory
    if user_input == 'no' or user_input == 'No':
        print("Will be using already existing trajectory!")
        with open(name + '_traj.pickle', 'rb') as f:
            myTraj = pickle.load(f)
            
    elif user_input != 'yes' or user_input != 'Yes':
        myTraj = createFullTrajectory(name, nLeafs, resolution, fov, readoutDur, gamma, gradMax, vkMax, slewMax, samplingRate, alpha, modulationType = modulationType, show = show, save = save)  
               
    return myTraj

def createFullTrajectory(name, nLeafs, resolution, fov, readoutDur, gamma, gradMax, vkMax, slewMax, samplingRate, alpha, modulationType ="centerOut", save = 0, show = 1):
    if show:
        # high quality images
        #set_matplotlib_formats('pdf', 'svg')
        
        # Set lower quality formats for faster plotting
        set_matplotlib_formats('png', 'jpg')
    
    """
    //////////////////////////////////////////
    Part 1: Making the trajectory
    """
    """
    a) Define trajectory parameters
    """
    myTraj = Yarnball(samplingRate, readoutDur, nLeafs, vkMax, resolution, fov, alpha, gradMax, slewMax, modulationType = modulationType)
    myTraj.name = name
    myTraj.calcParameters()
    
    """
    b) Optimize m for max discrepancy
    """    
    findOptimalM(myTraj, show = show) # will display graph of m vs discrepancy
    
    # save single interleave coordinates  for plotting
    #save('YBSingleInterleave.npy',myTraj.coordinates)

    """
    //////////////////////////////////////////
    Part 2: Checking the Garadient & Slew Waveforms 
    """
    # based on S. Vaziri and M. Lustig “The Fastest Gradient Waveforms” from: http://people.eecs.berkeley.edu/~mlustig/Software.html
    C, g, s, k, t, Nshift =  createGradients(myTraj, slewMax, show = 0)
    
    out = findOptimalM(myTraj, show = show, mOld = myTraj.m)
    changed = out[-1]
    # only update if the change is significant
    if changed == 1:
        C, g, s, k, t, Nshift =  createGradients(myTraj, slewMax, show = 0, repeat = True)
    
    # update coordinates and time values
    myTraj.coordinates = C    
    myTraj.tAll = t
    myTraj.g = g
    myTraj.s = s
    
    # caclulate index of images 
    Cr = np.linalg.norm(C,axis=1)
    localmax = np.r_[True, Cr[1:] > Cr[:-1]] & np.r_[Cr[:-1] > Cr[1:], True]
    
    if myTraj.nLeafs == 1:
        myTraj.localmaxIdx = [0, C.shape[0]-1] #localmax.nonzero()[0]
        #myTraj.localmaxIdx = localmax.nonzero()[0]
    else:
        localmax = localmax.nonzero()[0]
        localmax = localmax[-myTraj.nLeafs:]
        myTraj.localmaxIdx = np.insert(localmax[-myTraj.nLeafs:], [0], 0)
    
    if show == 1:
        """
        a) Plot single interleave
        """
        plotSingleInterleave(myTraj)
    
    # if surface trajectory (2D), end here
    if modulationType == 'none':
        return myTraj, []
    
    # not performing nyquist check for the R = 1 case for now
    myTraj.Nshots_R1 = None
    
    # save point around which the fibanocci rotations occur
    if nLeafs == 1:
        myTraj.rotPoint = myTraj.coordinates[-1,:]
    else:
        myTraj.rotPoint = myTraj.coordinates[myTraj.localmaxIdx[1],:]

    '''
    Part 4: Saving the info
    '''
    if save == 1:
        import pickle
        
        #saving traj
        with open(name + "_traj.pickle","wb") as f:
            pickle.dump(myTraj, f)
        
        # save unrotated coordinates (for Sajjad)
        if os.path.exists(name + '_g.dat'):
            os.remove(name + '_g.dat')
        if os.path.exists(name + '_v1.dat'):
            os.remove(name + '_v1.dat')
            
        if not(myTraj.nLeafs == 1) or modulationType =="edgeToEdge" : # only shift is multi-echo
            np.savetxt(name + '_Nshift.dat', [Nshift])
            
        np.savetxt(name + '_g.dat', g,  delimiter=",")
        
        np.savetxt(name + '_rotPoint.dat', myTraj.rotPoint, delimiter=",")
    
    if show == 1:
        """
        part 5: Density Assessment
        """
        # create 1000 rotations sample
        rotationPoint =  myTraj.coordinates[myTraj.localmaxIdx[1],:]
        allInterleaves, axisAngles, _ = FibanocciRotate(myTraj.coordinates, myTraj.res, 1000, v1 = rotationPoint)
        
        plotHistEachImage(allInterleaves, myTraj, top_color_bounds = 1)
            
        plotAllInterleaves(allInterleaves, myTraj, Nplot=5, echo=None, figure = True, tubeSize = 0.5)
        
        plotSurfaceSphere(allInterleaves, myTraj)
    return myTraj

if __name__ == '__main__': 
    
    #Input: #echos, resolution, fov, R, 
    nLeafs = 3 #echos
    resolution = 2e-3 # m
    fov = 240e-3 # m
    R = 1 #full sampling
    readoutDur = 6e-3 # s @7T (diff at other field strengths)
    
    #fixed: max v (max grad), max accel (slew rate), readout duration (40ms T2* decay @7T).
    gamma = 267.5153151e6 #rad/sT
    #gradMax = 40e-3 #T/m
    gradMax = 30e-3 #T/m
    vkMax = gradMax *gamma/(2*math.pi) # sT
    #slewMax = 180 #T/m/s
    slewMax = 180 #T/m/s
    samplingRate = 1e-6 # s (max)
    
    # radius modulation shape 
    alpha = 0.1
    #modulationType ="centerOut"
    modulationType ="edgeToEdge"
    save = 1
    show = 1
    
    name = giveName(nLeafs, fov, readoutDur, resolution, alpha, modulationType)
    
    myTraj = createFullTrajectory(name, nLeafs, resolution, fov, R, readoutDur, gamma, gradMax, vkMax, slewMax, samplingRate, alpha, modulationType = modulationType, save = save, show = show)
   
    print("Done: " + name)