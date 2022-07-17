from astropy.io import fits
from astropy.table import Table
import numpy as np
import pylab as plt
import random
from scipy import stats
from sklearn.neighbors import KDTree
import time
from sklearn.metrics import mean_squared_error
from astropy.cosmology import FlatLambdaCDM
from os import listdir
import scipy
import argparse

# which line to be used is set as an argument of the script. Line is important because there is a signal-to-noise cut on each line separately.
parser=argparse.ArgumentParser()
parser.add_argument('l', type=int)
args=parser.parse_args()

# parameters
n=30*10**3    # number of initial data points
nw=7781       # length of wavelength vector
run=0         # run is to keep track of which selection 
run_out=1     # run_out is to keep track of what I do to the spectra after selection
masking=True  # if true then emission lines will be masked+interpolated

lines=["OII_DOUBLET_EW","HGAMMA_EW","HBETA_EW","OIII_4959_EW","OIII_5007_EW","NII_6548_EW","HALPHA_EW","NII_6584_EW","SII_6716_EW","SII_6731_EW"]
lines_waves=[3728.5, 4342, 4862.7, 4960.3, 5008.2, 6549.9, 6564.6, 6585.3, 6718.3, 6732.7]  # vacuum wavelengths of the emission lines
for l in range(1): 
    l=args.l
    ## reading data
    spectra=np.load("/global/cscratch1/sd/ashodkh/results/spectra_selection"+str(run)+"_"+str(lines[l])+".txt.npz")["arr_0"]
    wavelengths=np.load("/global/cscratch1/sd/ashodkh/results/raw_data_wavelengths.txt.npz")["arr_0"]
    zs=np.load("/global/cscratch1/sd/ashodkh/results/zs_selection"+str(run)+"_"+str(lines[l])+".txt.npz")["arr_0"]
    
    # de-redshifting the wavelengths and finding the average lattice spacing of wavelength grid to use for interpolation
    nw=len(wavelengths)
    redshifted_waves=np.zeros([n,nw])
    for i in range(nw):
        redshifted_waves[:,i]=wavelengths[i]/(1+zs[:])
        
    d=np.average(redshifted_waves[:,1]-redshifted_waves[:,0])
    
    # setting up wavelength bins to calculate fluxes in them
    w1=3400
    w2=7000
    big_bin=np.arange(w1,w2,d)
    
    Ns=[6,11,16,21,26,31,41,51]
    for N in Ns:
        bin_ws=np.linspace(w1,w2,N)
        small_bins=[]
        for i in range(N-1):
            small_bins.append(np.arange(bin_ws[i],bin_ws[i+1],d))


        # calculating fluxes in bins for all the spectra
        c=3*10**18

        fluxes_bin=np.zeros([n,N-1])
        
        for i in range(n):
            if masking:  # masking emission lines
                spectrum_masked=np.zeros(spectra.shape[1])
                spectrum_masked[:]=spectra[i,:]
                for j in range(len(lines)):
                    waves_to_mask=np.where((redshifted_waves[i,:]>=lines_waves[j]-5)*(redshifted_waves[i,:]<=lines_waves[j]+5))[0]
                    x0=redshifted_waves[i, waves_to_mask[0]-1]
                    x1=redshifted_waves[i, waves_to_mask[-1]+1]
                    y0=spectra[i, waves_to_mask[0]-1]
                    y1=spectra[i, waves_to_mask[-1]+1]
                    spectrum_masked[waves_to_mask]=np.interp(redshifted_waves[i,waves_to_mask], [x0,x1], [y0,y1])
            for j in range(N-1):  # using nn interpolation to get spectrum on bin grids and then calculating flux
                x=redshifted_waves[i,:]
                tree=KDTree(x.reshape(-1,1))
                dist, ind=tree.query(small_bins[j].reshape(-1,1),k=1)
                if masking:
                    spectra_bin=spectrum_masked[ind].reshape(-1) 
                else:
                    spectra_bin=spectra[i,ind].reshape(-1)
                nans=np.where(np.isnan(spectra_bin[:]))[0]
                av=np.average(spectra_bin[spectra_bin==spectra_bin])
                spectra_bin[np.where(np.isnan(spectra_bin))[0]]=av
                fluxes_bin[i,j]=(10**23/3631)*np.trapz(10**(-17)*spectra_bin*small_bins[j]*(1+zs[i]),small_bins[j])\
                                /np.trapz(c/small_bins[j],small_bins[j])
            
        #np.savez_compressed("/global/cscratch1/sd/ashodkh/results/fluxes_bin_selection"+str(run_out)+"_"+str(lines[l])+"_bins"+str(N)+".txt",fluxes_bin)