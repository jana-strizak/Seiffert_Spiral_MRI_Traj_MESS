#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:18:56 2023

@author: janastri

version 1:normalization using qmc.scale 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import multiprocessing as mp
import pickle 
from math import sqrt
from copy import deepcopy
import matplotlib.cm as cm

def updateM(traj, m):
    trajNew = deepcopy(traj)
    trajNew.m = m
    trajNew.k = sqrt(m) #constant
    
    #ADDED 
    trajNew.make()
    return trajNew
    
# Function to find optimal m value by maximizing discrepancy 
def findOptimalM(traj, steps=100, mp_context=mp.get_context('forkserver'), show=0, image=0, mOld = None):
    # creating m to test 
    mTest = list(np.linspace(0,1,steps)) #= [0.2,0.5,0.8] #modulates shape of winding (determined by disperity optimization)
    #remove m = 0 and 1 (these values are not avaible for Joacobi's Eliptic functions)
    mTest = mTest[1:-1]
    
    pool = mp_context.Pool() if mp_context else mp.Pool()
    arg = [(traj, m) for m in mTest]
    
    l_bounds = [-traj.kmax, -traj.kmax, -traj.kmax]
    u_bounds = [traj.kmax, traj.kmax, traj.kmax]
    
    with pool as p: 
        # update M and remake trajectories
        Traj_list = p.starmap(updateM, arg) 
        
        # Normalizing
        nPointsLeaf = int(Traj_list[0].coordinates.shape[0]/traj.nLeafs)
        
        discrepancyAvrg= np.zeros([steps-2,traj.nLeafs])
        
        for i in range(traj.nLeafs):
            #coord = [traj.coordinates[nPointsLeaf*i:nPointsLeaf*(i+1),:] for traj in Traj_list]
            pointsNorm = [qmc.scale(t.coordinates[nPointsLeaf*i:nPointsLeaf*(i+1),:], l_bounds, u_bounds, reverse=True ) for t in Traj_list]
               
            # Calculating Discrepancy
            #DiscrepancyFunc = partial(qmc.discrepancy, method= 'WD')
            discrepancy = p.map(qmc.discrepancy, pointsNorm)
            
            discrepancyAvrg[:,i] = discrepancy
        
        #pointsNorm = DiaphanyCalc.normalize(Traj_list.coordinates)
    #iterating through all m values 
    #for i in range(mTest.shape[0]):
        #traj = Yarnball(mTest[i], traj.nPoints, traj.yarnLength, traj.nLeafs, traj.v_gradMax, modulationType ="centerOut")
        # remaking input trajectory with new m 
        #traj.updateM(mTest[i])
    
    #update 
            
        
        # Normalizing
        #pointsNorm = DiaphanyCalc.normalize(traj.coordinates)
        #l_bounds = [-1, -1]
        #u_bounds = [1.1, 1.1]

        #pointsNorm = qmc.scale(traj.coordinates, l_bounds, u_bounds, reverse=True)

        # Calculating Discrepancy
        #discrepancy = qmc.discrepancy(pointsNorm, method = 'WD')
        
        # Calculating Diaphnay
        #diaphany = DiaphanyCalc.Diaphany(pointsNorm)
        #allDiscrepancy.append(discrepancy)

        # Printing Results 
        #print('m = ', mTest[i], ' Discrepancy = ', discrepancy, ' Diaphany = ', diaphany)
        
        discrepancyAvrg = np.mean(discrepancyAvrg,axis=1)
        # select m based on smallest discrepency 
        i_mBest = np.argmin(discrepancyAvrg)
        
        mBest = mTest[i_mBest]
        
        print('mBest = ',str(mBest), ' with discrepency = ', discrepancyAvrg[i_mBest])
    
    discOld = []
    
    changed = 0
    if mOld != None: 
        shift = traj.coordinates.shape[0] - nPointsLeaf*traj.nLeafs
        cord = traj.coordinates[shift:, :]
        for i in range(traj.nLeafs):
            #cord = traj.coordinates[nPointsLeaf*i + Nshift : nPointsLeaf*(i+1),:].copy()
            cord_leaf = cord[nPointsLeaf*i : nPointsLeaf*(i+1),:]
            pointsNorm = qmc.scale(cord_leaf, l_bounds, u_bounds, reverse=True )
            discOld.append(qmc.discrepancy(pointsNorm))
        # averaged it
        discOld = sum(discOld)/len(discOld)
        
        print('old mBest = ',str(mOld), ' with current discrepency = ', discOld)
        
        d_diff = abs(discOld - discrepancyAvrg[i_mBest])
        
        if d_diff > 0.002: #
            print('changing m')
            #re-make with best m 
            traj.m = mBest
            traj.k = sqrt(mBest) #constant
            traj.make(show=0)   
            changed = 1
            mplot = mBest
            dplot = discrepancyAvrg[i_mBest]
        else:
            mplot = mOld
            dplot = discOld
    else:
        #re-make with best m 
        traj.m = mBest
        traj.k = sqrt(mBest) #constant
        traj.make(show=0)
        mplot = mBest
        dplot = discrepancyAvrg[i_mBest]
    
    if show: 
        plt.plot(mTest, discrepancyAvrg)
        plt.plot(mplot, dplot, marker="*")
        #plotting m and discrepancy 
        plt.ylabel('Discrepancy', fontsize=16)
        plt.xlabel('m', fontsize=16)
        #plt.text(r2 + 2, 0.5, 'FWHM:' + str(round(FWHM,3)) + 'mm', fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.show()
        
        stepSize = 10
        # create a sample of trajectories for plotting 
        #coordSample = coord[:i_mBest:stepSize] + coord[i_mBest::stepSize]
        #mTestSample = mTest[:i_mBest:stepSize] + mTest[i_mBest::stepSize]
        
        # saving data for plotting (single echo case)
        
        #if traj.nLeafs == 1:
         #   coordSample = [traj.coordinates[nPointsLeaf*image:nPointsLeaf*(image+1),:] for traj in Traj_list[:i_mBest:stepSize] + Traj_list[i_mBest::stepSize]]
        i_mWorst = np.argmax(discrepancyAvrg)
        mWorst = mTest[i_mWorst]
        print('mWorst = ',str(mWorst), ' with discrepency = ', discrepancyAvrg[i_mWorst])
         
        ## change this for single echo case 
        #coordSample = [traj.coordinates[nPointsLeaf*image:nPointsLeaf*(image+1),:] for traj in Traj_list[:i_mBest:stepSize] + Traj_list[i_mBest::stepSize]]
        #coordSample = [traj.coordinates[nPointsLeaf*image:nPointsLeaf*(image+1),:] for traj in [Traj_list[i_mBest], Traj_list[i_mWorst]]]
        coordSample = [traj.coordinates for traj in [Traj_list[i_mBest], Traj_list[i_mWorst]]]
        
        #mTestSample = mTest[:i_mBest:stepSize] + mTest[i_mBest::stepSize]
        mTestSample = [mplot, mWorst]
        i_mTestSample = [i_mBest, i_mWorst]
        
        return mplot, coordSample, mTestSample, mTest, i_mTestSample, discrepancyAvrg, changed
        
    return mplot, changed
'''
def plotM(samples, msample, kmax, nLeafs):
    # define subplot shifts 
    
    subplotColumns =len(samples)
    shift = 2*kmax
    middleVal = subplotColumns/2
    seperations = np.arange(0,subplotColumns)
    seperations = (seperations - middleVal)*shift
    
    fig =mlab.figure(bgcolor=(1,1,1), fgcolor=(0.,0.,0.))
    
    # Plotting the trajectories 
    for index in range(0, subplotColumns):
        c = samples[index]
        m = msample[index]
        # Making Color map
        colours = np.arange(nLeafs)
        pointsOneLeaf = c.shape[0]/nLeafs
        cMap = np.tile(colours,(int(pointsOneLeaf),1))
        cMap = np.reshape(cMap.T,cMap.shape[0]*cMap.shape[1])

        #mlab.plot3d(traj.coordinates[:,0], traj.coordinates[:,1], traj.coordinates[:,2], cMap, tube_radius=0.006, colormap = 'Accent')
        mlab.points3d(c[:,0] + seperations[index], c[:,1], c[:,2], scale_factor = 5)

        #plotting Background sphere
        #o = mlab.points3d(0, 0, 0, m++de='sphere', color=(1, 1, 1), scale_factor=2,opacity=0.2)
        #o.actor.property.interpolation = 'flat'
        theta1, phi1 = np.mgrid[0:np.pi:75j,-0.5*np.pi:1.5*np.pi:150j]
        r1 = kmax * np.ones_like(theta1)
        x1 = r1*np.sin(theta1)*np.cos(phi1)+seperations[index]
        y1 = r1*np.sin(theta1)*np.sin(phi1)
        z1 = r1*np.cos(theta1)
        mlab.mesh(x1,y1,z1,scalars=r1,opacity=0.1)
    
        #mlab.title('m = '+str(mTest[i]))
        label = 'm = '+str(round(m,2))
        mlab.text( seperations[index] - shift/4,  1*kmax + kmax/4, label, z=0, width=0.2)
        
    orientation_axes = mlab.orientation_axes()
    orientation_axes.axes.x_axis_caption_actor2d.caption_text_property.color = (0, 0, 0)
    orientation_axes.axes.y_axis_caption_actor2d.caption_text_property.color = (0, 0, 0)
    orientation_axes.axes.z_axis_caption_actor2d.caption_text_property.color = (0, 0, 0)
    
    mlab.view(0.0,0.0)
    fig.scene.parallel_projection = True
    return
 '''
    
if __name__ == "__main__":
    
    #mp_context = mp.get_context('forkserver')
    
    '''
    Importing Test Trajectroy
    '''
    with open('analyticalNfib_2Echos_FibRotate_traj.pickle', 'rb') as f:
        myTraj = pickle.load(f)
    
    mBest, coordSample, mTestSample = findOptimalM(myTraj,show=1)
    
    #plotSingleInterleave(myTraj)
    
    #mBest, coordSample = findOptimalM(sys.argv[1], show=1)
    """
    print
    if coordSample:
        #Serialize and save to file
        with open('coordSample.pkl', 'wb') as f:
            pickle.dump(coordSample, f)
    """
    
    plotM(coordSample, mTestSample, myTraj.kmax, myTraj.nLeafs)