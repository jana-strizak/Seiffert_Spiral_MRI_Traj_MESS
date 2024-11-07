#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# checking mech res and stot together
"""
Created on Thu May 11 14:28:33 2023

@author: janastri
"""
import pickle
import subprocess
import numpy as np
from SeiffertSpiralMRITraj.plottingFunctions import plotWaveform
from scipy.integrate import cumulative_trapezoid as cumtrapz
from math import pi, sqrt
import os
from SeiffertSpiralMRITraj.gradientMechanicalResonance import mechGradientResonance

def minTimeGradient(kSpaceCoord, g0, gfin, gmax, smax, T, show=1):
    gmax = gmax * 100 # [T/m] --> [G/cm]
    smax = smax /10 # [T/m/s] --> [G/cm/ms]
    #T = 1e-6 # s (max)
    T = T *1000 # [s] --> [ms]
    # changing percision to match double
    #kSpaceCoord = np.round(kSpaceCoord,17)/100 #converting to [1/cm]
    kSpaceCoord = kSpaceCoord/100 # to [1/cm]
    
    #save as data file
    file_path = 'temp/kspace.dat'

    np.savetxt(file_path, kSpaceCoord)

    file_out = 'temp/out.dat'
        
    # check if file is already compiled
    try:
        subprocess.check_output(['ls', 'SeiffertSpiralMRITraj/minTimeGradientJS'])
    except subprocess.CalledProcessError:
        # Compile the C file
        subprocess.run(['gcc', 'SeiffertSpiralMRITraj/minTimeGradientJS.c', '-o', 'SeiffertSpiralMRITraj/minTimeGradientJS','-lm'])
    
    # Run the executable
    #subprocess.run(['./minTimeGradient', 'kspace.dat', 'out.dat', '0', '0', '-1', '8', '20', '1e-3'])
    subprocess.run(['./SeiffertSpiralMRITraj/minTimeGradientJS', file_path, file_out, '0', str(g0), str(gfin), str(gmax), str(smax), str(T)], stdout=subprocess.DEVNULL)

    # load output file back in 
    with open(file_out, 'rb') as f:
        traj_out = np.loadtxt(f)
        
    # seperating the columns 
    C = traj_out[:,:3] *100 # reparamterized k-space [1/cm]-->[1/m]
    g = traj_out[:,3:6]/100 # gradients [G/cm] --> [T/m]
    s = traj_out[:,6:9] *10 # slew rates [G/cm/ms] --> [T/m/s]
    k = traj_out[:,9:12] *100 # real k-space from gradients [1/cm]-->[1/m] (This doesn't work well)
    
    t = traj_out[:,12]
    
    if os.path.exists(file_out):
        os.remove(file_out)
    if os.path.exists(file_path):
        os.remove(file_path)
    return C, g, s, k, t

# fine tune the arch length a little until timing is better 
def createGradients(traj, slewMax, show = 1, repeat = False):
    tTarget = (traj.tTot*1000)
    
    #inital arch length guess based on radius modulation and num of leafs
    if repeat:
        count = -4
    else:
        count = 0 
    result = 0
    g0, gfin = 0, -1
    
    #inital arch length guess based on radius modulation and num of leafs
    dg = 1/1000
    ds = 2
    gamma = 42.577478518*(10**6) #gyromagnetic ratio (Hz/T)

    # shifting to get gradients of multi-echo
    if show:
        print('/n')
    while result == 0:
        # current gradient and slew values 
        gcurr = traj.gradMax
        scurr = min( slewMax, traj.slewMax - count*ds)
        
        # update arclength total
        traj.s = np.linspace(0, traj.sTot, traj.nPoints_leaf*traj.nLeafs) # (1/m) modulates the length of the arch
        traj.s = traj.s/traj.kmax
        if show:
            #print("target arch-length:", round(traj.yarnLength*traj.nLeafs))
            print("target time:", round(traj.yarnLength*traj.nLeafs)/traj.vk*1000)
        # adjust s for proper timeing
        
        traj.make() # make the x,y,z coordinates of the trajectory
        
        # shifting to get gradients of multi-echo
        #shift = traj.coordinates[0,0]
        if not(traj.nLeafs == 1) or traj.modulationType =="edgeToEdge" : # only shift is multi-echo
            #traj.coordinates = traj.coordinates - shift
            endPoint = traj.coordinates[0,:]
            t_run = np.sqrt((1/gamma)*endPoint *(2/(traj.slewMax/2)))
            N_run = np.ceil(t_run / traj.sampleRate)
            g_blip = np.linspace(0, t_run*(traj.slewMax/2), num = int(N_run[0]))
            k_blip = gamma*np.cumsum(g_blip, axis = 0)*traj.sampleRate
            
            k_og = traj.coordinates
            traj.coordinates = np.vstack([k_blip,traj.coordinates])
            
        # check gradient timing
        C, g, s, k, t = minTimeGradient(traj.coordinates, g0, gfin, gcurr, scurr, traj.sampleRate)
        
        # remove blip time from readout time
        if not(traj.nLeafs == 1) or traj.modulationType =="edgeToEdge":
            # find how much needs to be shifted 
            C_round = np.round(C,1)
            Nshift = (C_round[:,0]!=C_round[:,1]).nonzero()[0][0] + 1 
            if Nshift%2 == 1: # odd not allowed 
                Nshift += 1
                
            t = t - t[Nshift]
        else:
            Nshift = 0

        resultMech = mechGradientResonance(g, gcurr, scurr, traj.sampleRate, show = show)
            
        # passing criteria
        if (tTarget - 0.1 < t[-1]) & (t[-1] <= tTarget + 0.1) & (resultMech == 0):
            result = 1 # pass
            
            traj.gradMax = gcurr
            traj.slewMax = scurr
            
            # shift back (only if multi-echo or starts at the periphery)
            if not(traj.nLeafs == 1) or traj.modulationType =="edgeToEdge":
                if show or result:
                    plotWaveform(t, C, title='K-Space Positions(including Blip)', ylabel='k position (1/m)')
                C = C[Nshift:, :]
                t_k = t[Nshift:]
            else:
                t_k = t
            if show or result: 
                print("total readout time for the entire trajectory:", t[-1])
                print('gcurr = ', gcurr)
                print('scurr = ', scurr)
                # plotting 
                plotWaveform(t_k, C, title='K-Space Positions', ylabel='k position (1/m)')
                plotWaveform(t, g, title='Gradients', ylabel='Gradient Magnitude (T/m)')
                plotWaveform(t, s, title='Slew Rate', ylabel='Slew Magnitude (T/m/s)')
         
        # timing doesn't pass
        if not((tTarget - 0.1 < t[-1]) & (t[-1] <= tTarget + 0.1)):
            # scale up or down sTot to improve result
            scale = tTarget / t[-1]
            #print('archlength changed from ' + str(traj.sTot) + ' to ', end = '')
            traj.sTot = scale * traj.sTot
            #print(str(traj.sTot))
            #count -= 10
        # gradient not passing
        if not(resultMech == 0):
            # change which g and slew used 
            count += 1
            
    return C, g, s, k, t, Nshift

if __name__ == "__main__":
    
    with open('analyticalNfib_1Echos_FibRotate_traj.pickle', 'rb') as f:
        myTraj = pickle.load(f)
    
    C, g, s, k, t = createGradients(myTraj)
