#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 12:43:00 2023

@author: janastri
"""

import numpy as np
import math 
import matplotlib.pyplot as plt
import scipy.special as sp
from SeiffertSpiralMRITraj.radiusModulation import radiusFunc
from scipy.stats import qmc
import multiprocessing as mp

class Yarnball:
    def __init__(self, sampleRate, tTot, nLeafs, vkMaxGrad, resolution, FOV, alpha, gradMax, slewMax, modulationType ="none", startTime = 0, m = None):
        self.m = m
        self.sampleRate = sampleRate
        #self.nPoints = nPoints  #points in one leaf
        self.tTot = tTot
        self.res = resolution
        self.FOV = FOV
        self.N = [int(FOV / resolution)] *3
        
        # extend kMax location for single echo so it meet nyquist parameter at kmax more easily 
        self.dk_nyquist = 1/FOV #dist between samples to meet nyquist criteria 
        if nLeafs == 1:
            self.kmax = 1/resolution/2 + self.dk_nyquist/2
        else:
            self.kmax = 1/resolution/2
        
        #self.yarnLength = yarnLength #for 1 interleave
        self.nLeafs = nLeafs
        #self.startPoint = startPoint
        #self.endPoint= endPoint
        self.vkMaxGrad = vkMaxGrad
        self.modulationType = modulationType
        if (nLeafs > 1) & (modulationType == 'centerOut'):
            self.modulationType = "edgeToEdge"
        self.startTime = startTime
        self.k = math.sqrt(m) if m else None #constant
        self.alpha = alpha
        self.gradMax = gradMax
        self.slewMax = slewMax

    def calcParameters(self):
        # determing velocity traveling and tmin 
        vk_nyquist = self.dk_nyquist/self.sampleRate # velocity in k to sample at k nyquist distance at each sampling rate time interval
        self.vk = min(self.vkMaxGrad, vk_nyquist) # check if it's less than the max gradient velocity 
        #vk = vkMaxGrad #delete this
        #print('vk = ', vk)
        
        # determining s arch
        self.nPoints_leaf = math.floor(self.tTot/self.nLeafs/self.sampleRate) #either defined by sampling rate or Kmin
        self.durLeaf = self.nPoints_leaf*self.sampleRate #actual duration based on whole num of points
        self.yarnLength = self.vk*self.durLeaf #changed so instead of tTot, it's slightly less depending on whole number of point
        
        #self.t = np.linspace( 0, self.tTot, self.nPoints_leaf*nLeafs) 
        #calculating time
        self.tLeaf =  np.linspace(self.startTime, self.durLeaf, self.nPoints_leaf) # time for 1 leaf
        self.tAll = np.linspace(self.startTime, self.durLeaf*self.nLeafs, self.nPoints_leaf*self.nLeafs) #time for multi-leaf
        #self.tTot = yarnLength/v_gradMax # aka: end time for 1 interleave
        
        # make arch length with guess 
        self.sGuess()  
        self.s = np.linspace(0, self.sTot, self.nPoints_leaf*self.nLeafs) # (1/m) modulates the length of the arch
        self.s = self.s/self.kmax
        return
    
    # function to determine archlength of trajectory based on radius modulation and mintimegrad function
    def sGuess(self):
        #determing radius modulation
        self.radiusModulation = radiusFunc(self.modulationType, self.alpha, self.tLeaf, self.nLeafs, self.kmax)
        
        # determining scale of final arch-length due to radius modulation 
        r_ratio = abs(self.radiusModulation)/self.kmax
        scaleDiff = np.shape(r_ratio)[0]/np.sum(r_ratio)
        
        # make arch-length -> s
        #self.s = np.linspace(0, self.yarnLength*self.nLeafs, self.nPoints_leaf*self.nLeafs) # (1/m) modulates the length of the arch
        #self.s = np.linspace(0, self.yarnLength*self.nLeafs*(scaleDiff*0.99), self.nPoints_leaf*self.nLeafs) # (1/m) modulates the length of the arch
        # the more leafs there are, the more the trajectory will be stretched to meet hardware constrains, so we will all a heuristic leaf scaling factor 
        leafScale = (1/self.nLeafs)**0.2
        self.sTot = self.yarnLength * self.nLeafs * (scaleDiff*0.9) * leafScale
        return 
    
    
    def make(self, show=0):
        # hyperbolic functions
        sn = sp.ellipj(self.s, self.m)[0]
        cn = sp.ellipj(self.s, self.m)[1]
        
        if show:
            #plotting radius
            plt.plot(self.tAll, self.radiusModulation)
            plt.title('Radius Modulation Function')
            plt.xlabel('Time')
            plt.ylabel('Radius')
            plt.show()
            
        # Make trjectroy in Cylindrical Coordinates
        theta = self.k*self.s
        r = sn
        z = cn
        
        #converting to cartesian
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = z
        coord =  np.vstack((x,y,z)).T
        
        v1 = np.array([x[0],y[0],z[0]])
        r1 =  np.linalg.norm(v1)
        
        v2 = np.array([1/math.sqrt(3),1/math.sqrt(3),1/math.sqrt(3)])
        r2 =  np.linalg.norm(v2)
        
        c = np.cross(v1, v2)
        rc = np.linalg.norm(c)
        d = np.dot(v1 ,v2)
        
        X = c / rc
        theta =  np.arccos( d / (r1 * r2))
        A = np.array([[ 0, -X[2], X[1]], [X[2], 0, -X[0]], [-X[1], X[0], 0]])
        R = np.eye(3) + np.sin(theta) * A + (1-np.cos(theta)) * A @ A
        
        # rotate by 45deg
        coordNew = (R @ coord.T).T
     
        xnew = self.radiusModulation * coordNew[:,0]
        ynew = self.radiusModulation * coordNew[:,1]
        znew = self.radiusModulation * coordNew[:,2]
        
        #self.coordinates = zip(x,y,z)
        self.coordinates =  np.vstack((xnew,ynew,znew)).T
        
        # calculating actual archlength 
        #diff_coord = np.linalg.norm(self.coordinates[1:,:] - self.coordinates[:-1,:],axis=1)
        #print("actual arch-length:", round(np.sum(diff_coord)))
        #print("actual time:", round(np.sum(diff_coord))/self.vk*1000)
        
        return self.coordinates
    
    '''
    # Function to find optimal m value by maximizing discrepancy 
    def findOptimalM(self, steps=100, mp_context=mp.get_context('forkserver'), show=0, image=0, mOld = None):
        # creating m to test 
        mTest = list(np.linspace(0,1,steps)) #= [0.2,0.5,0.8] #modulates shape of winding (determined by disperity optimization)
        #remove m = 0 and 1 (these values are not avaible for Joacobi's Eliptic functions)
        mTest = mTest[1:-1]
        
        pool = mp_context.Pool() if mp_context else mp.Pool()
        arg = [(self, m) for m in mTest]
        
        l_bounds = [-self.kmax, -self.kmax, -self.kmax]
        u_bounds = [self.kmax, self.kmax, self.kmax]
        
        with pool as p: 
            # update M and remake trajectories
            Traj_list = p.starmap(updateM, arg) 
            
            # Normalizing
            nPointsLeaf = int(Traj_list[0].coordinates.shape[0]/self.nLeafs)
            
            discrepancyAvrg= np.zeros([steps-2,self.nLeafs])
            
            for i in range(self.nLeafs):
                #coord = [traj.coordinates[nPointsLeaf*i:nPointsLeaf*(i+1),:] for traj in Traj_list]
                pointsNorm = [qmc.scale(t.coordinates[nPointsLeaf*i:nPointsLeaf*(i+1),:], l_bounds, u_bounds, reverse=True ) for t in Traj_list]
                   
                # Calculating Discrepancy
                discrepancy = p.map(qmc.discrepancy, pointsNorm)
                
                discrepancyAvrg[:,i] = discrepancy
   
            discrepancyAvrg = np.mean(discrepancyAvrg,axis=1)
            # select m based on smallest discrepency 
            i_mBest = np.argmin(discrepancyAvrg)
            
            mBest = mTest[i_mBest]
            
            print('mBest = ',str(mBest), ' with discrepency = ', discrepancyAvrg[i_mBest])
        
        discOld = []
        
        changed = 0
        if mOld != None: 
            shift = self.coordinates.shape[0] - nPointsLeaf * self.nLeafs
            cord = self.coordinates[shift:, :]
            for i in range(self.nLeafs):
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
                self.m = mBest
                self.k = math.sqrt(mBest) #constant
                self.make(show=0)   
                changed = 1
                mplot = mBest
                dplot = discrepancyAvrg[i_mBest]
            else:
                mplot = mOld
                dplot = discOld
        else:
            #re-make with best m 
            self.m = mBest
            self.k = math.sqrt(mBest) #constant
            self.make(show=0)
            mplot = mBest
            dplot = discrepancyAvrg[i_mBest]
        
        if show: 
            #plotting m and discrepancy 
            plt.plot(mTest, discrepancyAvrg)
            plt.plot(mplot, dplot, marker="*")
            plt.ylabel('Discrepancy', fontsize=16)
            plt.xlabel('m', fontsize=16)
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.show()

            i_mWorst = np.argmax(discrepancyAvrg)
            mWorst = mTest[i_mWorst]
            print('mWorst = ',str(mWorst), ' with discrepency = ', discrepancyAvrg[i_mWorst])
            coordSample = [traj.coordinates for traj in [Traj_list[i_mBest], Traj_list[i_mWorst]]]
            mTestSample = [mplot, mWorst]
            i_mTestSample = [i_mBest, i_mWorst]
            
            return mplot, coordSample, mTestSample, mTest, i_mTestSample, discrepancyAvrg, changed
        return mplot, changed         
    
    def plotSingleInter(): #might make plotting a function 
        return 
    
    def adjustGradients():
        return 
    
    def adjustSlew():
        return
    '''    
if __name__ == "__main__":
    
    #Input: #echos, resolution, fov, R, 
    nLeafs = 2 #echos
    resolution = 2e-3 # m
    fov = 256e-3 # m
    R = 1 #full sampling
    readoutDur = 5e-3 # s @7T (diff at other field strengths)
    
    #fixed: max v (max grad), max accel (slew rate), readout duration (40ms T2* decay @7T).
    gamma = 267.5153151e6 #rad/sT
    gradMax = 30e-3 #T/m
    vkMax = gradMax *gamma/(2*math.pi) # sT
    slewMax = 140 #T/m/s
    samplingRate = 1e-6 # s (max)
    
    # radius modulation shape 
    alpha = 0.5

    showPlots = 1 # will display single multiple interleaves
    NinterleavesPlot = 5 # choose how many interleaves will be displayed, leave 'None' for all
    
    myTraj = Yarnball(samplingRate, readoutDur, nLeafs, vkMax, resolution, fov, alpha, gradMax, slewMax, modulationType ="centerOut")
    myTraj.calcParameters()
    
    # fake m determination 
    myTraj.m = 1
    myTraj.k = math.sqrt(myTraj.m) #constant
    
    #ADDED 
    myTraj.make()
    
    # fine tuning Stot
    myTraj.findStot()