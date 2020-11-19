#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import argrelmax
import glob
import rasterio
import rasterio.features
import rasterio.warp
#from pyproj import Proj, transform
#from osgeo import gdal, osr

"""
Created on Thu Sep  5 08:33:33 2019

@author: sfons

Functions to read in DMS imagery and classify it using a histogram mode
detection method. The classififcation looks for leads to use in FB calculation from 
OIB ATM and snow  / Ku-band radars.

The main function (classify) is broken up into indiculat functions for calling
in the example jupyter notebook.
"""


dmsdir = '/Volumes/icebridgedata/IODMS1B_DMSgeotiff_v01/2011_AN_NASA/20111013/'


def get_dms_files(dms_dir):
    """
    Returns a list of dms files from the given directory

    """
    return glob.glob(dms_dir+'*.tif')
    

def band_hist(band,order=10):
    """
    Gets a histagram of the provided band, as well as max/min thresholds for 
    classifying.
    Input:
        DMS Band image
        Order of argrelamax (=7) for peak finding
    Output:
        histogram
        bin_edges
        maxima
        minima
    
    """
    hist, edges = np.histogram(band.ravel(), bins=255, range=[0,1],density=True)

    
    #find modes of histogram
    #order of 7 means it looks at 7 points on either side of peak. 
    #Determined empirically to be pretty robust.
    maxima = argrelmax(hist,order=order) 
    maxima = maxima[0] #locations of modes
    
    #if theres a false last max, remove it
    if hist[maxima[-1]] < 1:
        maxima = maxima[0:-1]
        
    #find minima between modes
    minima = []
    for ind,mx in enumerate(maxima[:-1]):
        minimum = np.argmin(hist[mx:maxima[ind+1]])
        minima.append(minimum + mx)
    minima = np.asarray(minima)
    
    return hist,edges, maxima, minima


def class_from_hist(red,minima,maxima):
    '''
    Function for jupyter notebook. Part of classify() below.
    '''
    # filter out pixels based on first minimum (they are boarder remnants)
    red[red < (minima[0]/255)] = np.nan
    
    # classify image
    red_classed = red.copy()
    
    # first, if there are two modes w/in 20brightness values, use second last
    lastmin = minima[-1]
    if (maxima[-1] - maxima[-2]) < 20:
        #lastmax = maxima[-2]
        lastmin = minima[-2]
        
    # ice is everything greater than last minimum
    red_classed[red >= lastmin/255] = 3 #3 is ice
    
    # if more maxima than two, segment further
    if len(minima) == 2: #3 maxima
        red_classed[(red >= np.double(minima[0])/255) & (red < 0.2)] = 1 #1 is open water
        red_classed[(red < np.double(lastmin)/255) & (red >= 0.2)] = 2 #2 is young ice
    if len(minima) == 3: #4 maxima
        red_classed[(red >= np.double(minima[0])/255) & (red < np.double(minima[1])/255) ] = 1 #1 is open water
        red_classed[(red < np.double(lastmin)/255) & (red >= np.double(minima[1])/255) ] = 2 #2 is young ice 
    if len(minima) > 3:
        red_classed[(red >= np.double(minima[0])/255) & (red < 0.2)] = 1 #1 is open water
        red_classed[(red < np.double(lastmin)/255) & (red >= 0.2)] = 2 #2 is young ice

    # make unit, remove nans
    red_classed[np.isnan(red_classed) == 1] = 0
    red_classed = np.uint16(red_classed)
    
    return red_classed

def classify(fnames,output_dir):
    """
    Classifies DMS images using a red band histogram thresholding method, 
    similar to that put forth in Buckley et al. 2020. 
    4 Surface types are chosen:
        0. Border pixels
        1. Open Water
        2. Young Sea Ice
        3. Snow-covered Sea Ice
        
    Input:
        list of filenames from get_dms_files
        Output directory
    Output:    
        Saves classfied image to specified directory.
    """
    
    for fname in fnames: #use for giving certain filenames
        fname_short = fname[:-4]
        

        # Using rasterio 
        ds = rasterio.open(fname)
        red = ds.read(1)
        red = np.double(red)
        red[red == 0] = np.nan
        red = red/255
        
        # Using GDAL
        '''
        ds = gdal.Open(fname)
        data = ds.ReadAsArray()
        #multiband tif
        data = data[:3, :, :].transpose((1, 2, 0)) #reshapes! only for multiband tif
        red = data[:,:,0]
        red = np.double(red)
        red[red == 0] = np.nan
        red = red/255
        #singleband
        #rb = data.copy()
        gt = ds.GetGeoTransform()
        '''
        
        # Get histogram
        hist,bins,maxima,minima = band_hist(red)

    
        # filter out pixels based on first minimum (they are boarder remnants)
        red[red < (minima[0]/255)] = np.nan
        
        # classify image
        red_classed = red.copy()
        
        # first, if there are two modes w/in 20brightness values, use second last
        lastmin = minima[-1]
        if (maxima[-1] - maxima[-2]) < 20:
            #lastmax = maxima[-2]
            lastmin = minima[-2]
            
        # ice is everything greater than last minimum
        red_classed[red >= lastmin/255] = 3 #3 is ice
        
        # if more maxima than two, segment further
        if len(minima) == 2: #3 maxima
            red_classed[(red >= np.double(minima[0])/255) & (red < 0.2)] = 1 #1 is open water
            red_classed[(red < np.double(lastmin)/255) & (red >= 0.2)] = 2 #2 is young ice
        if len(minima) == 3: #4 maxima
            red_classed[(red >= np.double(minima[0])/255) & (red < np.double(minima[1])/255) ] = 1 #1 is open water
            red_classed[(red < np.double(lastmin)/255) & (red >= np.double(minima[1])/255) ] = 2 #2 is young ice 
        if len(minima) > 3:
            red_classed[(red >= np.double(minima[0])/255) & (red < 0.2)] = 1 #1 is open water
            red_classed[(red < np.double(lastmin)/255) & (red >= 0.2)] = 2 #2 is young ice
    
        # make unit, remove nans
        red_classed[np.isnan(red_classed) == 1] = 0
        red_classed = np.uint16(red_classed)
        
        #SAVING
        with rasterio.open(
            output_dir+fname_short+'_class.tif',
            'w',
            driver='GTiff',
            height=red_classed.shape[0],
            width=red_classed.shape[1],
            count=1,
            dtype=red_classed.dtype,
            crs=ds.crs, #'EPSG:4326',
            transform=ds.transform,
        ) as dst:
            dst.write(red_classed, 1)
    return 

