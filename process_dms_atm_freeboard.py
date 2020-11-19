#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelmax
import glob
import rasterio
from pyproj import Proj, transform
#import cartopy.crs as ccrs
import pandas as pd
import pickle
from osgeo import gdal

"""
Created on Thu Sep 12 14:01:40 2019

@author: Steven Fons
         sfons@umd.edu

This script labels Operation Icebridge ATM data using coincident DMS images. 
The process is as follows:
    1) read in dms image and classify surface types:
        [Classes: Snow-covered ice, young ice, open water, n/a]
    2) reconcile CRS and projections between ATM and DMS
    3) Align/overlay ATM data with DMS imagery
    4) Label atm points according to their surface type
    5) save a .pkl file of the labelled ATM data
    
"""

def main(atm_floc,dms_floc):

    date = '20111013'
    atm_loc = '/Volumes/icebridgedata/ILATM1B_ATMqfit_v01/2011_AN_NASA/'+date+'/'
    atm_files_df = pd.DataFrame()
    atm_files_df['f_list'] = glob.glob(atm_loc+'ILATM1B_'+date+'*.qi')
    atm_files_df['time'] = [int(a[-17:-11]) for a in atm_files_df['f_list']]
    atm_files_df.sort_values('time')
    atm_files_df = atm_files_df[atm_files_df['time'] >= 165111] #for 2011 flight, time needs to be after 165111 (overlap)
    
    dms_loc = '/Volumes/icebridgedata/IODMS1B_DMSgeotiff_v01/2011_AN_NASA/'+date+'/'
    out_loc = atm_loc+'/test_class/'
    
    # loop through atm files
    for ind,afile in enumerate(atm_files_df['f_list']):
        
        print('Processing file '+str(ind)+' of '+str(len(atm_files_df['f_list'])))
        short_fname = afile[-34:-3] #gets just the name of the file
        
        # read in atm data from binary file
        atm_data = read_atm(afile)
        
        # get list of dms images within each file
        dms_df = get_dms_list(atm_files_df,ind, dms_loc,date)
        print(str(len(dms_df['f_list']))+' matching DMS images. Processing...')
        
        # loop through DMS images w/in atm files
        atm_df = pd.DataFrame()
        for ind,img in enumerate(dms_df['f_list']):
            
            # classify dms image
            dms_class, gt = classify_dms(img)
            
            # classify atm data and concat for all DMS images!
            temp_atm = atm_data.copy()
            atm_df_t = classify_atm(temp_atm,dms_class)
            atm_df = pd.concat([atm_df,atm_df_t],ignore_index=True)
            
        
        # Drop un-needed columns from DF
        atm_df_save = atm_df.copy()
        atm_df_save = atm_df_save.drop(columns=['y_reproj', 'x_reproj','x_pix','y_pix','dms_val'])
        
        # Save ATM df (classed) as pkl file! LL,Elev,Class,Time
        with open(out_loc+short_fname+'_classified.pkl','wb') as f:
            pickle.dump([atm_df_save],f)
            
    
        
    return print('Processing Complete!')

        

def read_atm(atm_file):
    '''
    This function reads in a binary atm .qi file
    and returns a pandas dataframe of the output.
    '''
    with open(atm_file,mode='rb') as file:
        data_array = np.fromfile(file, np.int32).reshape(-1,12)
    data_array = np.double(data_array)
    data_array[:,1] = data_array[:,1] / 1000000 #lat
    data_array[:,2] = data_array[:,2] / 1000000 #lon
    data_array[:,3] = data_array[:,3] / 1000 #elevation
    data_array= data_array[84:,:]
    df = pd.DataFrame()
    df['time'] = data_array[:,0]
    df['lat'] = data_array[:,1]
    df['lon'] = data_array[:,2]
    df['elev'] = data_array[:,3]
    return df
    

def get_dms_list(atm_df,ind,dms_loc,date):
    '''
    This function returns a df list of DMS images that are taken during a
    single ATM file, along track
    Provide a list of atm files, since it requires the time of the next atm file 
    to subset DMS images
    '''

    # get time of ATM (and ATM + 1) from filestring for DMS images
    try:
        trange = [atm_df['time'][ind],atm_df['time'][ind+1]]
    except:
        # IF last file, add X amount of min/sec
        trange = [atm_df['time'][ind],atm_df['time'][ind]+1000] #1000 is 10 minutes. 
        
    # get DMS images that fall within that time range
    dms_df = pd.DataFrame()
    dms_df['f_list'] = glob.glob(dms_loc+'DMS_*'+date+'*.tif')
    dms_df['time'] = [int(d[-12:-6]) for d in dms_df['f_list']] #with class? [-18:-12]
    dms_df = dms_df[dms_df['time'] >= trange[0]]
    dms_df = dms_df[dms_df['time'] <= trange[1]]
    dms_df.sort_values('time')
    return dms_df


def classify_dms(img_file):
    '''
    reads in a DMS image filename, and classifies it into 1 of 4 categories:
        0) N/A or boarder pixels
        1) open water / leads
        2) grey/young ice
        3) Snow covered sea ice

    Following a red-band histogram-mode classification similar to that used in
    Buckley et al. 2020
    
    Returns:
        a classified, uint16 2d matrix AND the gdal transform information
        (used in next step to classify ATM data)
    '''
    
    
    ds = gdal.Open(img_file)
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

    # get histogram of red
    histo, edges = np.histogram(red.ravel(), bins=255, range=[0,1],normed=True)
    #x = np.linspace(0, 1, num=255) #for plotting
    
    # find modes of histo
    maxima = argrelmax(histo,order=7) #order of 10 means it looks at 10 points on either side of peak. Pretty robust.
    maxima = maxima[0] #locations of modes
    
    # if theres a false last max, remove it
    if histo[maxima[-1]] < 1:
        maxima = maxima[0:-1]
    # find minima between modes
    minima = []
    for ind,mx in enumerate(maxima[:-1]):
        minimum = np.argmin(histo[mx:maxima[ind+1]])
        minima.append(minimum + mx)
    minima = np.asarray(minima)
    
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
    red_classed[red >= lastmin/255] = 3 #3 is ice, for now.
    
    # if more maxima than two, segment further
    if len(minima) == 2:
        red_classed[(red >= np.double(minima[0])/255) & (red < 0.2)] = 1 #1 is open water
        red_classed[(red < np.double(lastmin)/255) & (red >= 0.2)] = 2 #2 is young ice
    if len(minima) == 3:
        red_classed[(red >= np.double(minima[0])/255) & (red < np.double(minima[1])/255) & (red < 0.2)] = 1 #1 is open water
        red_classed[(red < np.double(lastmin)/255) & (red >= np.double(minima[1])/255) & (red >= 0.2)] = 2 #2 is young ice
    if len(minima) > 3:
        red_classed[(red >= np.double(minima[0])/255) & (red < 0.2)] = 1 #1 is open water
        red_classed[(red < np.double(lastmin)/255) & (red >= 0.2)] = 2 #2 is young ice

    # make uint, remove nans
    red_classed[np.isnan(red_classed) == 1] = 0
    red_classed = np.uint16(red_classed)
    
    return red_classed, gt
    

def classify_atm(atm_df,dms_class,gt):
    '''
    This script takes in ATM data, a classified DMS image, and DMS geographic
    transform information to classify ATM data shots as leads, young ice, or
    snow-covered ice.
    Output: 
        a dataframe of relevant ATM information
    '''
    inProj = Proj(init='epsg:4326')
    outProj = Proj(init='epsg:3976')
    x1,y1 = np.asarray(atm_df['lon']),np.asarray(atm_df['lat'])
    x2,y2 = transform(inProj,outProj,x1,y1)
    atm_df['y_reproj'] = y2
    atm_df['x_reproj'] = x2
    
    # Removes all ATM points that fall outside of DMS image
    atm_df = atm_df[atm_df['x_reproj'] > gt[0]]
    atm_df = atm_df[atm_df['y_reproj'] < gt[3]]
    
    # Get dms pixel coordiantes from the reprojected ATM coordinates
    px = ((atm_df['x_reproj'] - gt[0]) / gt[1]).astype(int) #x pixel
    py = ((atm_df['y_reproj'] - gt[3]) / gt[5]).astype(int) #y pixel
    # DMS images are flipped, so flip x and y
    pix_x = py
    pix_y = px
    
    # Removes all points that (Again?) fall outside the image
    pix_x[pix_x > np.shape(dms_class)[0]-1] = np.nan
    pix_y[pix_y > np.shape(dms_class)[1]-1] = np.nan
    atm_df['x_pix'] = pix_x
    atm_df['y_pix'] = pix_y
    atm_df = atm_df.dropna()
    
    # Get dms values at ATM point, add to df
    dms_val = dms_class[np.asarray(atm_df['x_pix']).astype(int),np.asarray(atm_df['y_pix']).astype(int)]
    atm_df['dms_val'] = dms_val
    atm_df = atm_df[atm_df['dms_val'] != 0]
    
    # classify!! lead=1,ice=2,snow=3, NAN = 0. #not totally necessary but nicer to look at
    atm_df['class'] = np.zeros((len(atm_df['lat'])))
    atm_df.loc[atm_df['dms_val'] == 1, 'class'] = 'lead' 
    atm_df.loc[atm_df['dms_val'] == 2, 'class'] = 'ice' 
    atm_df.loc[atm_df['dms_val'] == 3, 'class'] = 'snow' 
    atm_df.loc[atm_df['dms_val'] == 0, 'class'] = 'none' 
    
    return atm_df

if __name__ == '__main__':
    main()