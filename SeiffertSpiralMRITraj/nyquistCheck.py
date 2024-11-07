#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:15:46 2023
Notes:
    Attempting nyquist test between NBH interleaves to find number of fibanocci points to meeting minimum nyquist distance
@author: janastri

updated for mulltiecho case
got rid of check points and instead fibanocci rotates the entire thing 

checking nyquist condition using cartesian grid to make sure no holes 

update oct 19th: 
    1) checks that each point and  it's neighbours have at least 1 point inside
    and 
    2) the max(dist_min(i,ngb)) = K_nyquist  
    
    also includes: random sampling of grid points to check 
"""
import SeiffertSpiralMRITraj.nyquistTest as nyquistTest
from SeiffertSpiralMRITraj.addingRotationsManually import addRotationsHere
# for inital guess
from SeiffertSpiralMRITraj.analyticalNfib import findNfib as findInitalGuess

import numpy as np
import os
import math
from SeiffertSpiralMRITraj.FibanocciLatticeRotation import FibanocciRotate
import pickle
from time import time
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def getNfibStart(myTraj, Nfib, R):
    if Nfib == None:
        Nfib = findInitalGuess(myTraj)

        if R < 2:
            Nfib = int(Nfib * 2)
            
        elif R < 3:
            Nfib = int(Nfib * 0.2)
            
        elif R < 6:
            Nfib = int(Nfib * 1.5 / (R**1.5))
            
        elif R < 12:
             Nfib = int(Nfib * 1 / (R**1.5)) 
             
        elif R < 20:
            Nfib = int(Nfib * 0.4 / (R**1.5))
    return Nfib

def findNfibNyquist(myTraj, show = 0, R = 1, save = 0, name = None, Nfib = None):
    # nyquist criteria 
    R = (R)**(1/3)
    nyquistDist = myTraj.dk_nyquist * R
    
    # display matrix search size 
    print("Matrix size :" , int(myTraj.FOV / R / myTraj.res), " X ", int(myTraj.FOV / R / myTraj.res), " X ", int(myTraj.FOV / R / myTraj.res))
    
    # multi-echo case 
    if myTraj.nLeafs == 1:
        coordinates = myTraj.coordinates
    else:
        idx = 0
        startPointImage = myTraj.localmaxIdx[idx]
        endPointImage = myTraj.localmaxIdx[idx+1]
        coordinates = myTraj.coordinates[startPointImage:endPointImage,:]
        
    # inital guess
    Nfib = getNfibStart(myTraj, Nfib, R)
    
    
    # make search grid (from -kmax to kmax with )
    x_ =  np.linspace(-myTraj.kmax, myTraj.kmax, num = math.floor(2*myTraj.kmax/nyquistDist)+1)
    y_ =  np.linspace(-myTraj.kmax, myTraj.kmax, num = math.floor(2*myTraj.kmax/nyquistDist)+1)
    z_ =  np.linspace(-myTraj.kmax, myTraj.kmax, num = math.floor(2*myTraj.kmax/nyquistDist)+1)

    xSearch, ySearch, zSearch = np.meshgrid(x_, y_, z_, indexing='ij')
    
    # radius at each point 
    d = np.sqrt(xSearch**2 + ySearch**2 + zSearch**2)
    
    # only acceptible search space 
    dsphere = np.zeros((d.shape), dtype = np.int8)
    
    # core points to conduct grid search on (labeled with a 2)
    mask_core = (d <=  myTraj.kmax - nyquistDist/4) & (d > myTraj.kmax/5)
    dsphere[mask_core.nonzero()[0], mask_core.nonzero()[1], mask_core.nonzero()[2]] = 2
    
    # exlcude inner radius 
    r = np.linalg.norm(coordinates, axis = 1)

    exclude_inner = (r >=  myTraj.kmax/5).nonzero()[0]
    coordinates_outter = coordinates[exclude_inner,:]
    
    gridPoints = np.stack([xSearch, ySearch, zSearch],axis=-1)  #(dim Dx X Dy X Dz X 3)
    gridPoints = np.array(gridPoints, dtype = np.float32)
    
    # rearrange to 2D
    pointsSearch = np.vstack([xSearch.flatten(), ySearch.flatten(), zSearch.flatten()]).T
    # this can be reversed with xSearch_f = xSearch_f.reshape(130,130,130)
    
    print( " total num of grid points to search through: ", len(dsphere.nonzero()[0]))

    cont = True
    
    manualShots = False
    newShots = None
    
    Lold = None
    L = 1
    NfibOld = None
    
    Rall = None 
    
    # for plotting total 
    NfibAll = []
    NmissingAll = []
    
    while (cont):
        twhile = time()
        
        if not(manualShots):
            # make all shot rotations
            checkPoints_allInter, axisRotations, _ = FibanocciRotate(coordinates_outter, myTraj.res, Nfib, v1 = coordinates_outter[-1,:])
        else:
            #myTraj.axisAngles = axisRotations
            
            if myTraj.nLeafs == 1:
                checkPoints_allInter = np.dstack([checkPoints_allInter, newShots[exclude_inner,:]])
            else:
            #  combine those for first echo only (to check again)
                newShots_image = newShots[startPointImage:endPointImage,:]
                checkPoints_allInter = np.dstack([checkPoints_allInter, newShots_image[exclude_inner,:]])
            Nfib = checkPoints_allInter.shape[2]
        
        # rotate only first image for search
        print('Nfib: ', Nfib)
        
        NfibAll.append(Nfib)
        
        # reshape to 2D
        pointsTraj2D = np.reshape(np.transpose(checkPoints_allInter,[2,0,1]), [checkPoints_allInter.shape[0]*checkPoints_allInter.shape[2], checkPoints_allInter.shape[1]])
        pointsTraj2D = np.array(pointsTraj2D, dtype = np.float32)
        
        # make KD tree with all traj points
        kdtree = cKDTree(pointsTraj2D)

        # get num of CPU for the computer 
        Ncpu = os.cpu_count()
        
        # points within grid point's nyquist sphere
        pointsNyquist = kdtree.query_ball_point(pointsSearch, r=nyquistDist, workers = Ncpu)
        pointsNyquist = np.array([pointsTraj2D[i] for i in pointsNyquist], dtype=object)
        # reshape to 3D
        kdtree = None 
        
        pointsNyquist = pointsNyquist.reshape(gridPoints.shape[0],gridPoints.shape[1],gridPoints.shape[2])

            
        # delete this line(only for troubleshooting):
        #pointsNyquistOG = deepcopy(pointsNyquist)
        
        # length operation for matrices 
        mylen = np.vectorize(len)
        len_ny = mylen(pointsNyquist)
        
        # determine which grid points are empty (if they are empty, they fail nyquist test)
        empty_list_mask = len_ny == 0 #np.array([len(item) == 0 for item in pointsNyquist])
        # make all empty gridpoints 2
        mask_insphere_empty = np.logical_and(mask_core, empty_list_mask)
        
        NnotpassingAll = np.shape(mask_insphere_empty.nonzero()[0])[0]
        print(" completely empty grid points: ", NnotpassingAll )
        
        # update dsphere to not check already failing points 
        
        # all the points with empty cores that need a trajector passing through them
        emptyCore = gridPoints[mask_insphere_empty.nonzero()[0], mask_insphere_empty.nonzero()[1], mask_insphere_empty.nonzero()[2],:]
        
        # setting empty values to zero in the sphere 
        dsphere[mask_insphere_empty.nonzero()[0], mask_insphere_empty.nonzero()[1], mask_insphere_empty.nonzero()[2]] = 0
        
        # mask out grid points with too many nyquist points 
        maxPointsthreshold = 10000
        # find where there are more points in one square than anywhere else 
        mask_tooManyPoints = len_ny > maxPointsthreshold
        # those points will just pass (not checked, same as if it were outside the sphere with dsphere value of 0)
        dsphere[mask_tooManyPoints.nonzero()[0], mask_tooManyPoints.nonzero()[1], mask_tooManyPoints.nonzero()[2]] = 0
        
        # making empty list to populate it with 
        lists = np.empty((len(mask_tooManyPoints.nonzero()[0])), dtype=object)
        lists[:] = [[] for _ in range(len(lists))]
        
        pointsNyquist[mask_tooManyPoints.nonzero()[0], mask_tooManyPoints.nonzero()[1], mask_tooManyPoints.nonzero()[2]] = lists

        # performing 2 step nyquist test 
        MissingPoints = nyquistTest.process_3d_array(gridPoints, pointsNyquist, dsphere, nyquistDist)
        
        MissingPoints = np.array(MissingPoints)
        
        if NnotpassingAll > 0 and len(MissingPoints) > 0:
            MissingPoints = np.concatenate((emptyCore, MissingPoints))
            NmissingAll.append(MissingPoints)
        elif NnotpassingAll > 0 and len(MissingPoints) == 0:
            MissingPoints = emptyCore
        
        if len(MissingPoints) > 0:
            MissingPoints = np.unique(MissingPoints, axis = 0)

        # num of missing points
        NumMissingPoints = MissingPoints.shape[0]
        percentageMissing = round(NumMissingPoints/Nfib*100)
            
        print("Num of non-passing points = ", NnotpassingAll)
        print("Percentage of non-passing points = ", round((NnotpassingAll/pointsSearch.shape[0])*100), "%")
        print("num of missing points = ", NumMissingPoints)
        print("percent of missing points = (missinngPoints/Nfib) ", round(NumMissingPoints/Nfib*100),"%")
        print('time to conduct this Nfib check = '+ str(round(time()-twhile,2)))
        
        # if fail, must change 
        tooSmall = (NumMissingPoints > 0 and percentageMissing > 11)
        tooLarge = (not(manualShots) and percentageMissing < 5)
        
        if tooSmall or tooLarge:
            print("not enough rotations")
            
            if not(Lold == None) and not(Lold == L):
                
                #gradient decent method
                L = 9 - percentageMissing
                dL = L - Lold
                dNfib = Nfib - NfibOld
                
                Lold = L 
                NfibOld = Nfib
                
                Nfib = math.ceil(Nfib - (3)*(dL/dNfib))
                
            elif tooSmall: # update first time
                NfibOld = Nfib
                Lold = NumMissingPoints
                Nfib =  math.ceil(Nfib * ( 1 + 6 * (NumMissingPoints/len(mask_core.nonzero()[0]))))
            elif tooLarge:
                Nfib = math.floor(0.9*Nfib)
                
        elif NumMissingPoints > 0 and percentageMissing <= 11:
            # addRotationsHere(myTraj, MissingPoints, pointsSearch)
            print("Performing Manual Rotations")
            
            if MissingPoints.ndim == 1:
                MissingPoints = np.expand_dims(MissingPoints, axis = 0)
                
            #newShots, Rall = addRotationsHere(myTraj, MissingPoints)
            newShots, Rall = addRotationsHere(coordinates, myTraj.coordinates, MissingPoints)
            newShots = np.stack(newShots, axis = 2)
            
            # save manually created rotation matrices 
            manualShots = True
        else: # done
            cont = False 
    
    # end of while loop (Nfib has been found)
    
    del pointsNyquist # free space
    
    # finish rotating other interleaves
    # rotate the rest of the points since nyquist is met 
    coordinatesAll = myTraj.coordinates
    coordinatesAll_allInter, _, _ = FibanocciRotate(coordinatesAll, myTraj.res, axisRotations.shape[0], v1 = coordinates[-1,:], axisAngles = axisRotations)
    #coordinatesAllImages = np.vstack([checkPoints_allInter_fib, coordinatesAll_allInter])
    
    # join with manual shots
    kloc_complete = np.dstack([coordinatesAll_allInter, newShots])
    
    # saving R = 1 case for future undersampling
    if R == 1:
        myTraj.Nshots_R1 = Nfib
        # save traj with NShots, axis angles, and rotation matrices for extra shots
        traj_path = os.path.abspath(os.path.join(myTraj.name, '..'))
        traj_name = os.path.basename(myTraj.name)
        with open(traj_path + '/' + traj_name + '_traj.pickle', 'wb') as f:
            pickle.dump(myTraj, f)
    
    # update traj info 
    myTraj.R = R
    myTraj.Nshots = Nfib
    myTraj.axisAngles = axisRotations
    myTraj.Rall = Rall
    myTraj.kspaceCoord = kloc_complete
    myTraj.name = name 
    
    if save:
        # saving
        with open(name + '_traj.pickle', 'wb') as f:
            pickle.dump(myTraj, f)
            
        np.save(name + "_allInterleaves_nyTest", kloc_complete)
        
        if myTraj.nLeafs > 1:# multi-echp
            for i in range(myTraj.nLeafs-1):
                echoCurr = kloc_complete[myTraj.localmaxIdx[i]:myTraj.localmaxIdx[i+1],:,:]
                np.save(name + "_echo" + str(i) + '_nyTest', echoCurr)
                echoCurr2D = np.reshape(np.transpose(echoCurr,[2,0,1]), [echoCurr.shape[0]*echoCurr.shape[2], echoCurr.shape[1]])
                np.savetxt(name + "_echo" + str(i) + '_allInterleaves2D_nyTest' + '.dat', echoCurr2D)
            i = myTraj.nLeafs - 1
            echoCurr = kloc_complete[myTraj.localmaxIdx[i]:,:,:]
            np.save(name + "_echo" + str(i) + '_nyTest', echoCurr)
            echoCurr2D = np.reshape(np.transpose(echoCurr,[2,0,1]), [echoCurr.shape[0]*echoCurr.shape[2], echoCurr.shape[1]])
            np.savetxt(name + "_echo" + str(i) + '_allInterleaves2D_nyTest' + '.dat', echoCurr2D)
        
        # save axis rotations 
        np.savetxt(name + '_allInterleaves_nyTest' + '_axisAngles.dat', myTraj.axisAngles,  delimiter=",")
        # save rotation matrices 
        if Rall is not None:
            flattened_matrices = [matrix.flatten() for matrix in Rall]
            np.savetxt(name + '_RExtraShots_nyTest.txt', np.array(flattened_matrices), delimiter=',', fmt='%s')
            np.savetxt(name + '_RExtraShots_SajjadFormat_nyTest.txt', np.array(flattened_matrices), delimiter=',', fmt='%s', newline='},\n{')
        
    return myTraj


if __name__ == "__main__":
    freeze_support()
    
    """
    Importing a Trajectroy
    """  
    name = "/export02/data/Jana/NewSpiral/YB_1Echos_240FOV_4msRO_1p5mmRes_1alpha/YB_1Echos_240FOV_4msRO_1p5mmRes_1alpha"
    with open(name + "_traj.pickle","rb") as f:
        myTraj = pickle.load(f) 
        
    r = 2
    
    nameNew = name.split('/')[-1] + "_R" + str(r)
    
    #making r directoory
    directory = ''.join([s + '/' for s in name.split('/')[:-1]])
    RDir = directory + 'R' + str(r)
    try:
        os.makedirs(RDir)
    except:
        None
    
    nameNew = RDir + '/' + nameNew 
    
    # performing a nyquist check
    t = time()
    NfibBest, point_allInter, Rall = findNfibNyquist(myTraj, R = r, name = nameNew, save = 1)
    
    
    print('time to conduct Nyquist check:', str(round(time()-t,2)),'s')
    
    print("Done")
    