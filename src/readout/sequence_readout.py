import os
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import yaml
import cv2
from skimage.io import imread
from skimage.io import imsave
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "pgf.texsystem": "xelatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'figure.dpi': 300,
})
from skimage.feature import peak_local_max
import math

TOPHAT_KERNEL_SIZE = 7
def tophat_spots(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(TOPHAT_KERNEL_SIZE,TOPHAT_KERNEL_SIZE))
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

def extract_coordinates(image, snr=4, quantile=0.96):
    meta = {}
    coordinates = peak_local_max(image,min_distance=2,threshold_abs=snr*np.mean(image))
    meta['Coordinates brighter than given SNR'] = coordinates.shape[0]
    meta['Image mean intensity'] = float(np.mean(image))
    intensities = image[coordinates[:,0],coordinates[:,1]]
    meta[f'{quantile} quantile'] = float(np.quantile(intensities,quantile))
    threshold = np.quantile(intensities,quantile)
    coordinates = coordinates[image[coordinates[:,0],coordinates[:,1]]>threshold]
    meta['Final spots count'] = coordinates.shape[0]
    return coordinates,meta

def crop_coordinates(coordinates,x_start,y_start,x_width,y_width):
    ll = np.array([y_start,x_start])
    ur = ll + np.array([y_width,x_width])
    coordinates_crop = coordinates[np.all(np.logical_not(np.logical_and(ll<=coordinates,coordinates<ur)),axis=1)]
    return coordinates_crop

channels = ['cy3','cy5']
#stc_directory = './stitched'
def get_intensity_df(stc_directory,coordinates,thresholds,tophat=False):
    #stc_directory = os.path.join(in_directory,'stitched')
    intensity_df = pd.DataFrame({'Y':coordinates[:,0],'X':coordinates[:,1]})
    for cyc in range(1,10+1):
        for channel in channels:
            # im = tophat_spots(imread(os.path.join(stc_directory,f'cyc_{cyc}_{channel}.tif')))
            im = imread(os.path.join(stc_directory,f'cyc_{cyc}_{channel}.tif'))
            im[im>50000] = 0
            if tophat:
                im = tophat_spots(im)
            # im = cv2.dilate(im,np.ones((3,3),dtype=np.uint16))
            intensity_df[f'cyc_{cyc}_{channel}'] = im[coordinates[:,0],coordinates[:,1]]
    #intensity_df = intensity_df[intensity_df.iloc[:,2:].max(axis=1)>=min(thresholds.values())]
    return intensity_df

#thresholds = {'cy3': 500,'cy5':1000}
def get_seq_df_old(intensity_df,coordinates,thresholds):
    bool_df = pd.DataFrame({'Y':coordinates[:,0],'X':coordinates[:,1]})
    for cyc in tqdm(range(1,10+1),desc='Boolean thresholding'):
        for channel in channels:
            bool_df[f'cyc_{cyc}_{channel}'] = intensity_df[f'cyc_{cyc}_{channel}']>=thresholds[channel]# *decay[i]
    base_bool_map = {(True,True):'A',(True,False):'T',(False,True):'C',(False,False):'G'}
    base_df = pd.DataFrame({'Y':coordinates[:,0],'X':coordinates[:,1]})
    for cyc in tqdm(range(1,1+10),desc='Base calling'):
        base_df[f'cyc_{cyc}'] = list(zip(bool_df[f'cyc_{cyc}_cy3'],bool_df[f'cyc_{cyc}_cy5']))
        base_df[f'cyc_{cyc}'] = base_df[f'cyc_{cyc}'].apply(lambda x: base_bool_map[x])
    seq_df = pd.DataFrame({'Y':coordinates[:,0],'X':coordinates[:,1]})
    seq_df['Sequence'] = base_df[[f'cyc_{i}' for i in range(1,1+10)]].agg(''.join,axis=1)
    return seq_df

def get_base(x,y):
    if x == y == True:
        return 'A'
    elif x == True and y == False:
        return 'T'
    elif x == False and y == True:
        return 'C'
    else:
        return 'G'
    
def get_seq_df(intensity_df,thresholds):
    coordinates = intensity_df[['Y','X']].to_numpy()
    bool_df = pd.DataFrame({'Y':coordinates[:,0],'X':coordinates[:,1]})
    for cyc in range(1,10+1):
        for channel in channels:
            bool_df[f'cyc_{cyc}_{channel}'] = intensity_df[f'cyc_{cyc}_{channel}']>=thresholds[channel]# *decay[i]
    base_bool_map = {(True,True):'A',(True,False):'T',(False,True):'C',(False,False):'G'}
    base_df = pd.DataFrame({'Y':coordinates[:,0],'X':coordinates[:,1]})
    for cyc in tqdm(range(1,1+10),desc='Base calling'):
        base_df[f'cyc_{cyc}'] = bool_df.apply(lambda x: get_base(x[f'cyc_{cyc}_cy3'],x[f'cyc_{cyc}_cy5']),axis=1)
    seq_df = pd.DataFrame({'Y':coordinates[:,0],'X':coordinates[:,1]})
    seq_df['Sequence'] = base_df[[f'cyc_{i}' for i in range(1,1+10)]].agg(''.join,axis=1)
    return seq_df

def factorize_intensity_df(intensity_df):
    df_factorized = pd.DataFrame()
    for i in range(1,11):
        for channel in channels:
            test_df = intensity_df.loc[:,['Y','X',f'cyc_{i}_{channel}']]
            test_df.rename(columns={f'cyc_{i}_{channel}':'Intensity'},inplace=True)
            test_df['Cycle'] = i
            test_df['Channel'] = channel
            if channel == 'cy3':
                test_df['Intensity'] = test_df['Intensity'] * 2
            df_factorized = pd.concat([df_factorized,test_df])
    return df_factorized

def boxplot_intensities(intensity_df,seq_df,title,iteration,ax):
    intensity_df_merged = intensity_df.merge(seq_df,how='inner')
    df_factorized = factorize_intensity_df(intensity_df_merged)
    sns.boxplot(ax=ax[iteration],x='Cycle',y='Intensity',hue='Channel',data=df_factorized)
    ax[iteration].set(ylim=(0,50000))
    ax[iteration].set_title(title)

def overlap(seq_x,seq_y):
    output = ''
    for x,y in zip(seq_x,seq_y):
        if x == y:
            output += x
        elif 'G' not in x+y:
            output += 'A'
        else:
            if 'C' in x+y:
                output += 'C'
            elif 'T' in x+y:
                output += 'T'
    return output

def get_overlap_list(seq,seq_list):
    overlap_list = [seq]
    for s in seq_list:
        if s != seq:
            overlap_list.append(overlap(s,seq))
    return overlap_list

def draw_spots_localized(df,size,name):
    radius = size // 2
    coordinates = df[['Y','X']].to_numpy()
    y_count = math.ceil(coordinates.shape[0]**0.5)
    x_count = coordinates.shape[0] // y_count + 1
    for i in range(1,11):
        for channel in channels:
            canvas = np.zeros((y_count*size,x_count*size),dtype=np.uint16)
            im = imread(f'./stitched/cyc_{i}_{channel}.tif')
            for j in range(coordinates.shape[0]):
                y,x = coordinates[j,:]
                y_index = j // x_count
                x_index = j % x_count
                canvas[y_index*size:(y_index+1)*size,x_index*size:(x_index+1)*size] = im[y-radius:y+radius+1,x-radius:x+radius+1]
            try:
                os.makedirs(os.path.join('.','error_spots',name))
            except FileExistsError:
                pass
            imsave(os.path.join('.','error_spots',name,f'cyc_{i}_{channel}.tif'),canvas)
