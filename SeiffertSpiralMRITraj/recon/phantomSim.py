import numpy as np 
import matplotlib.pyplot as plt
# BART method
from SeiffertSpiralMRITraj.recon.imageRecon import performRecon, save_as_nifti
from SeiffertSpiralMRITraj.utility import readcfl, writecfl
import os
import subprocess
 
# import trajectory to be used 
def makePhantom(kloc, nCoils, FOV, savename_dir, formatKloc = True):
    savename = savename_dir + "/phantomKspace_" + str(nCoils) + "Coils"
    
    # check if the phantom already exists
    if os.path.exists(savename + ".npy"):
        print(f"A file with the name '{savename}' exists.")
        # read the kspace file
        kspace = np.load(savename + ".npy")
    else:
        # format kloc
        if formatKloc:
            kloc = np.reshape(np.transpose(kloc,[2,0,1]), [kloc.shape[0]*kloc.shape[2], kloc.shape[1]])
            kloc = kloc.T * FOV
            kloc = np.array(kloc, dtype = np.float32)
        writecfl('temp/kloc', kloc)
        
        kspace_output_name = 'temp/phantomKspace'
        # make 3D phantom in kspace domain with 4 virtual coils 
        #!bart phantom -3 -t kloc_R2 -s 4 -k phantom_kspace
        
        # make 3D phantom on Bart 
        print("creating phantom...")
        bash_command = f"bart phantom -3 -t temp/kloc -s {nCoils} -k {kspace_output_name}"
        out = subprocess.run(bash_command, shell=True, capture_output=True, text=True)
        if out.returncode != 0:
            error_message = out.stderr.strip()
            raise Exception(f"Error executing bash command: {error_message}")
        kspace = readcfl(kspace_output_name).squeeze()
        
        # save as npy for future use and for viewing, since computing this takes a long time
        np.save(savename + '.npy', kspace)
        
        # delete kloc from temp folder 
        if os.path.exists('temp/kloc' + '.cfl'):
            os.remove('temp/kloc' + '.cfl')
            os.remove('temp/kloc' + '.hdr')
            
        # delete kspace cfl and hdr files
        if os.path.exists(kspace_output_name + ".cfl"):
            os.remove(kspace_output_name + ".hdr")
            os.remove(kspace_output_name + ".cfl")

    return kspace # default name

def groundTruthPhantom(N):
    phantomGT_name = 'temp/phantomGT'
    bash_command = f"bart phantom -3 -x {N[0]} {phantomGT_name}"
    out = subprocess.run(bash_command, shell=True, capture_output=True, text=True)
    if out.returncode != 0:
        error_message = out.stderr.strip()
        raise Exception(f"Error executing bash command: {error_message}")
    phantomGT = readcfl(phantomGT_name).squeeze()
    
    # delete phantom ground truth 
    if os.path.exists(phantomGT_name + ".cfl"):
        os.remove(phantomGT_name + ".cfl")
        os.remove(phantomGT_name + ".hdr")        
    return phantomGT

def phantomSimImage(traj, Ncoils = 6, reconIter = 50, lamda = 0.01, save = True, show = True):
    # make sim folder to save in
    saveDir = os.path.dirname(traj.name) + '/sim'
    
    if save:
        if not os.path.exists(saveDir): # Check if the directory does not exist
            os.makedirs(saveDir)
            print(f"Directory '{saveDir}' created.")
        else:
            print(f"Directory '{saveDir}' already exists.")

    outputName = saveDir + "/phantomImage_" + str(Ncoils) + "Coils"
    
    # check if simulation already exists
    user_input = 'yes'
    if os.path.exists(outputName):
        user_input = input("This simulated phantom: " + outputName + " already exists. Do you want to remake the simulation? (yes/no): ").strip().lower()
    if user_input == "YES" or user_input == "yes" or user_input == "Yes":
        # get matrix size 
        N = traj.N
        
        # make the ground truth phantom to compare with
        phantomGT = groundTruthPhantom(N)
        
        # format kspace coordinate locations 
        kloc = traj.kspaceCoord
        kloc = np.reshape(np.transpose(kloc,[2,0,1]), [kloc.shape[0]*kloc.shape[2], kloc.shape[1]])
        kloc = kloc.T * traj.FOV
        kloc = np.array(kloc, dtype = np.float32)
        
        # get kspace of phantom sampled along the non-cartesian trajectory's locations
        phantomK = makePhantom(kloc, Ncoils, traj.FOV, saveDir, formatKloc=False)
        
        # reconstruct phantom image sampled along non-cartesian k-space coordinates
        image, _ = performRecon(phantomK, kloc.T, max_iter=reconIter, N = N, eigen_iter = 20, lamda=lamda)
        
        if save:
            # save image as nifti for better viewing
            save_as_nifti(image, outputName)
        
        if show:
            # plot ground truth 
            plotImage(phantomGT, plotTitle = "Ground Truth Phantom")
            # plot image
            plotImage(image, save = save, saveName = outputName, plotTitle = "Simulated Sampling Along Trajectory Phantom")
    # output simulated image and ground truth 
    return image, phantomGT

def plotImage(Image, scale=1, save = False, saveName = None, plotTitle = None):
    # plot original phantom
    maxVal = np.max(abs(Image))
    
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(abs(Image[:,:,int(Image.shape[2] * 0.4)]), cmap="grey", interpolation = 'none', vmax = maxVal*scale)
    axes[0].axis('off') 
    axes[1].imshow(abs(Image[:,int(Image.shape[1] * 0.45),:]), cmap="grey", interpolation = 'none', vmax = maxVal*scale)
    axes[1].axis('off') 
    axes[2].imshow(abs(Image[int(Image.shape[0] * 0.57),:,:]), cmap="grey", interpolation = 'none', vmax = maxVal*scale)
    axes[2].axis('off')
    if plotTitle:
        plt.suptitle(plotTitle)
    plt.tight_layout()
    plt.subplots_adjust(top=1.3)
    plt.show()
    
    if save:
        plt.savefig(saveName, format = 'tif')
    return