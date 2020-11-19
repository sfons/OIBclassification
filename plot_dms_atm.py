#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from osgeo import gdal
#import glob
#from matplotlib.colors import LinearSegmentedColormap
"""
Created on Tue Sep 10 15:10:21 2019

@author: steven fons (sfons@umd.edu)
This program reads in a DMS image and ATM data, re projects them to a common
projection, and plots them both.

"""

##  Read in DMS Image (RGB or Classified)
dmsfile = '/Users/sfons/research/presentations_conferences/AGU2019/presentation/DMS_1142610_03840_20110326_13492663.tif'
dmsfilee = '/Users/sfons/research/presentations_conferences/AGU2019/presentation/DMS_1142610_03840_20110326_13492663_classified.tif'
#dms_fname = '/Volumes/icebridgedata/IODMS1B_DMSgeotiff_v01/2011_GR_NASA/20110326/DMS_1142610_03571_20110326_13440385.tif'
#dms_list = glob.glob(dms_fname+'DMS_*'+date+'*.tif')
##  Read in ATM data
atmfile = '/Users/sfons/research/presentations_conferences/AGU2019/presentation/ILATM1B_20110326_134720.ATM4BT4.qi'

def plot_atm_dms(atmfile,dmsfile):
    
    """
    Function that overlays ATM elevation data on top of a DMS image from 
    Operation IceBridge. This function reprojects the datasets to a common 
    projection first using gdal.
    **Only plots data, doesn't find overlapping**
    
    Input:
        atmfile: File path of ATM data (.qi files)
        dmsfile: File path of DMS image (classified or raw)
    Output:
        Plot of ATM elevation and DMS image below
    """
    
    # load in dms file
    ds = gdal.Open(dmsfile)
    data = ds.ReadAsArray()
    gt = ds.GetGeoTransform()
    
    # get image extent based on DMS image
    extent = (gt[0], gt[0] + ds.RasterXSize * gt[1],gt[3] + ds.RasterYSize * gt[5], gt[3])

    with open(atmfile,mode='rb') as file:
        data_array = np.fromfile(file, np.int32).reshape(-1,12)
    data_array = np.double(data_array)
    data_array[:,1] = data_array[:,1] / 1000000 #lat
    data_array[:,2] = data_array[:,2] / 1000000 #lon
    data_array[:,3] = data_array[:,3] / 1000 #elevation
    data_array= data_array[84:,:] #trims header info


    # Plot Data
    #define projection to plot in
    projection_crs = ccrs.epsg(3413) #3412 = NSIDC south polar stereo, 3413 OIB NH, 3031 = SH
    #create figure and plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1,projection=projection_crs)
    
    # plot ATM data first
    scat = ax.scatter(data_array[:,2], data_array[:,1],s=3,c=data_array[:,3]-21,marker='o',cmap='jet',vmax=4,alpha=0.31,transform=ccrs.Geodetic())
    clrbar = plt.colorbar(scat,orientation="horizontal",fraction=0.046, pad=0.04)
    clrbar.set_alpha(.85)
    clrbar.set_label('Elevation [m]')
    clrbar.draw_all()
    
    # plot DMS image
    if len(np.shape(data)) == 2: #only one channel / classification images
        implot = data.copy()
        v_max = 6
    elif len(np.shape(data)) == 3: #3 channels / RGB image
        implot = data[:3, :, :].transpose((1, 2, 0)) #reshapes! 
        implot = implot[:,:,0] #to only use 1st band
        v_max = 255
        
    # make cmap if using classified data
    '''
    colors = ['black', 'white','darkgray','darkgray','black','lightgray']  # R -> G -> B
    n_bins = 6  # Discretizes the interpolation into bins
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    '''
    
    overlap_plot = ax.imshow(implot, cmap='gray',vmin=0,vmax=v_max,origin='upper', extent=extent, transform=ccrs.epsg(3413))
    
    return overlap_plot

overlap_plot = plot_atm_dms(atmfile, dmsfile)