#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 19:08:08 2023

@author: janastri
"""
import numpy as np
from matplotlib import pyplot as plt

def checkPeaks(f, ft2, bw, fb):
    result = 0
    
    # global max
    maxGlobal = np.argmax(ft2)
    
    # looping through all frequencies and bandwidth
    for i in range(0, len(fb)):
        if (fb[i] - bw[i]/2 - 5) <= f[maxGlobal] <= (fb[i] + bw[i]/2 + 5):
            # failed
            result = 1
    return result
    
def mechGradientResonance(g, gMax, sMax, dt, show = 1):
    result = 0
    
    # Terra Forbidden Frequ
    #fb = [1100, 550, 367];
    #fb = [1000, 550];
    fb = [590, 1140];
    
    #bw = [300, 100, 22];
    bw = [100, 200];
    
    if show:
        # plot in spyder
        fig, axs = plt.subplots(figsize=(7, 6))
        
        for i in range(0,np.shape(fb)[0]):
            for k in range(-int(bw[i]/2),int(bw[i]/2)):
                plt.plot([fb[i] + k, fb[i] + k], [0,1], color = "r")

    oversample = 4;
    L = np.shape(g)[0]*oversample;
    L = L - L%2;
    #dt = 2e-6;
    
    directionLabels = ["x", "y", "z"]
    
    #plt.figure()
    for i in range(0, np.shape(g)[1]):
        # fourier transform of gradient waveforms 
        ft = np.fft.fftshift(abs(np.fft.fft(g[:,i],n = L)));
        ft2 = ft[int(L/2):]
        ft2 = ft2/max(ft2)
        f  = np.linspace(0, 1/dt/2, int(L/2))
        df = f[1]-f[0];
        
        # checking if peaks are within forbidden band
        result = result + checkPeaks(f, ft2, bw, fb)
        if show:
            plt.plot(f, ft2, label = directionLabels[i])
    if show:
        plt.xlabel('Frequency (Hz)', fontsize=18)
        plt.legend(fontsize=18)
        plt.xlim([0, 1500])
        plt.title(f"Gmax = {gMax}, SlewMax = {sMax}", fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        #plt.legend()
        plt.show()
    '''
    plt.figure()
    for i in range(0, np.shape(g)[1]):
        # fourier transform of gradient waveforms 
        ft = np.fft.fftshift(abs(np.fft.fft(g[:,i],n = L)));
        ft2 = ft[int(L/2):]

        f  = np.linspace(0,1/dt/2,int(L/2))
        df = f[1]-f[0];
        
        # integral 
        ft2_int = np.trapz(ft2)
        print("integral: %f"%ft2_int)
        plt.plot(f, ft2/ft2_int, label = directionLabels[i])
    
    plt.xlabel('Frequency (Hz)')
    plt.legend()
    plt.xlim([0, 1500])
    plt.show()
    '''
    return result

if __name__ == "__main__":
    g = np.loadtxt("/data_/tardiflab/Jana/NewSpiral/g1.dat")
    
    result = mechGradientResonance(g)
    
    print(result)