#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:14:55 2024

@author: janastri
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.interpolate import UnivariateSpline
from SeiffertSpiralMRITraj.recon.imageRecon import performRecon

def makePSF(traj, save = 0, T2_star = 0, decayTime = 33.2, echoRO = 0, EchoNumber = 0):
    kloc = traj.kspaceCoord
    savename = traj.name + "_PSF"
    
    if T2_star:
        #EchoNumber = 0
        #echoRO = 6e-3 #[s]
        savename += "withT2starDecay"
        
        Nshots = kloc.shape[2]
        Npoints = kloc.shape[0]
        dt = 1e-6 #[s]
        
        time = dt * np.linspace( 0, Npoints, Npoints)
        time = np.tile(time, Nshots)
        
        T2star_greyMatter = decayTime/1000 #[s]
        kspace = 1*np.exp(-(time + (EchoNumber) * echoRO)/T2star_greyMatter).T
        kspace = np.array(kspace, dtype = np.complex64)
        #kspace = np.reshape(kspace, [1, kspace.shape[0]])
        
    kloc = np.reshape(np.transpose(kloc,[2,0,1]), [kloc.shape[0]*kloc.shape[2], kloc.shape[1]])
    kloc = kloc * traj.FOV
    kloc = np.array(kloc, dtype = np.float32)
    
    if not(T2_star):
        kspace = np.ones(kloc.shape[:-1], dtype=np.complex64)
       
    #return kspace
    # oversampling kspace
    OSFactor = 2
    kloc2 = kloc * OSFactor # doesn't add new memory
    #kloc2 = kloc  # doesn't add new memory
    
    img_shape = [int(traj.FOV/traj.res)] * 3
    img2_shape = [i * OSFactor for i in img_shape]
    
    psf, _ = performRecon(kspace, kloc2, max_iter=30, N = img2_shape, eigen_iter = 6, psf = 1)
    plotPSF(psf, traj.FOV, traj.res, savename, save = save)
    
    #psf = nufft_adjoint(kspace, kloc2, oshape=img2_shape ,oversamp=1.25) # adds memory
    #plotPSF(psf, traj.FOV, traj.res, savename, save = save)
    
    if save:
        np.save(savename, psf)
        
    return kspace

def plotPSF(psf, FOV, res, savename, save = 0):
    #np.save(savename + "_PSF", psf)
    # find peak position of psf
    xmaxI, ymaxI, zmaxI = np.unravel_index(np.argmax(abs(psf), axis=None), psf.shape)
    
    #X = np.arange(-psf.shape[0]/2, psf.shape[0]/2)
    #Y = np.arange(-psf.shape[1]/2, psf.shape[1]/2)
    kmax = int(FOV / res / 2) 
    
    X = np.linspace(-kmax, kmax, psf.shape[0])
    Y = np.linspace(-kmax, kmax, psf.shape[1])
    X, Y = np.meshgrid(X, Y)
    
    psf_plot =  [abs(psf[:,:,zmaxI]) / np.max(abs(psf)), abs(psf[:,ymaxI,:]) / np.max(abs(psf)), abs(psf[xmaxI,:,:]) / np.max(abs(psf))]
    titles = ["XY", "XZ", "YZ"]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={"projection": "3d"})
    
    for i in range(len(titles)):
        ax = axs[i]
        surf = ax.plot_surface(X, Y, psf_plot[i], cmap=cm.jet, alpha=1, antialiased=False, vmin = 0, vmax = 0.08, rstride=1, cstride=1)
        ax.set_title( titles[i] + " Plane", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        # Add a color bar which maps values to colors.
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.09)
    cbar = fig.colorbar(surf, ax=axs, shrink=0.6, aspect=15)
    cbar.ax.tick_params(labelsize=16)

    # save
    if save:
        plt.savefig(savename + "_planes.tiff", format = 'tiff')
    plt.show()
    
    # calculating point FWHM 
    psf_plot =  [abs(psf[:,ymaxI,zmaxI]) / np.max(abs(psf)), abs(psf[xmaxI,:,zmaxI]) / np.max(abs(psf)), abs(psf[xmaxI,ymaxI,:]) / np.max(abs(psf))]
    titles = ["X", "Y", "Z"]
    
    fig, axs = plt.subplots( 1, 3, figsize=(13, 4))
    
    for i in range(len(titles)):
        ax = axs[i]
        
        # Half FWHM 
        xplot = np.arange(-psf.shape[0]/2, psf.shape[0]/2) * res*1000 # (res / (psf.shape[0] / (FOV / res )))
        xplot_i = xplot[round(xplot.shape[0]*(3/8)):round(xplot.shape[0]*(5/8))]
        psf_plot_i = psf_plot[i][round(xplot.shape[0]*(3/8)):round(xplot.shape[0]*(5/8))]
        #spline = UnivariateSpline(xplot, psf_plot[i]-np.max(psf_plot[i])/2, s=0)
        spline = UnivariateSpline(xplot_i, psf_plot_i - np.max(psf_plot_i)/2, s = 0)
        
        #x_values = np.linspace(xplot[round(xplot.shape[0]*(3/8))], xplot[round(xplot.shape[0]*(5/8))], 100)
        #y_values = spline(x_values)
        r1, r2 = spline.roots() # find the roots
        FWHM = r2 - r1
        
        # plot 
        ax.plot( xplot, psf_plot[i])
        ax.set_title(titles[i] + ' Axis', fontsize=18)
        ax.set_xlabel(titles[i] + ' (mm)', fontsize=16)
        ax.text(r2, 0.5, 'FWHM:' + str(round(FWHM,2)) + 'mm', fontsize=16)
        ax.scatter(r1, 0.5, color='red', marker='.') 
        ax.scatter(r2, 0.5, color='red', marker='.') 
        ax.set_xlim(-int(xplot.max()/12), int(xplot.max()/12))
        ax.tick_params(axis='both', which='major', labelsize=16)
        
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.85, wspace=0.15)
    
    if save:
        plt.savefig(savename + "_Axis.tiff", format = 'tiff')
    
    plt.show()
    
    return