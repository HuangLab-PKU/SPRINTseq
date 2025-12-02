import random

from numpy.core.defchararray import index
from sequence_readout import extract_coordinates
from sequence_readout import get_intensity_df
from sequence_readout import get_seq_df
from sequence_readout import tophat_spots
from reference_check import check_sequence
from mapping import map_barcode
from mapping import unstack_plex
import os
import sys
from datetime import datetime
from tqdm import tqdm
from tqdm import trange
import numpy as np
from skimage.io import imread
from skimage.io import imsave
import pandas as pd

CHANNELS = ['cy3','cy5']
CYCLE_NUM = 4
BASE_DIRECTORY = r'\\10.10.10 Images Archive'
BASE_DEST_DIRECTORY = r'\\10.10.10.1\NAS Processed Images'
THRESHOLDS = {'cy3':500,'cy5':700} # 500， 700
SNRS = {'cy3':1.0,'cy5':1.0}
RAW_SEQ_NAME = 'raw_sequence.csv'
CHECKED_NAME = 'ref_checked.csv'
SEQ_CYCLE = 10
QUANTILE = 0.1

def get_coordinates(in_directory,channels=CHANNELS,cycle_num=CYCLE_NUM):
    coordinates = None
    for i in trange(1,1+cycle_num):
        for channel in channels:
            im = imread(os.path.join(in_directory,f'cyc_{i}_{channel}.tif'))
            im[im>50000] = 0
            if not np.any(coordinates):
                coordinates,_ = extract_coordinates(im,snr=SNRS[channel],quantile=QUANTILE)
            else:
                temp,_ = extract_coordinates(im,snr=SNRS[channel],quantile=QUANTILE)
                coordinates = np.unique(np.concatenate((coordinates,temp)),axis=0)
    return coordinates

def extract_raw_sequence(in_directory,out_directory,thresholds,output_image=True):
    coordinates = get_coordinates(in_directory)
    print(f'Extracted {coordinates.shape[0]} puncta.')
    intensity_df = get_intensity_df(in_directory,coordinates,tophat=True,cyc_num=SEQ_CYCLE)
    min_threshold = min(thresholds.values())
    intensity_df = intensity_df[intensity_df.iloc[:,2:].max(axis=1)>=min_threshold]
    intensity_df.to_csv(os.path.join(out_directory,'intensity_filtered.csv'),index=False)
    print(f'Obtained intensity sequence of {len(intensity_df)} puncta.')
    intensity_df=pd.read_csv(os.path.join(out_directory,'intensity_filtered.csv'))
    seq_df = get_seq_df(intensity_df,thresholds,cyc_num=SEQ_CYCLE)
    print(f'Obtained raw sequence dataframe.')
    seq_df.to_csv(os.path.join(out_directory,RAW_SEQ_NAME))

def main(run_id):
    #run_id = sys.argv[1]
    #ref_file = sys.argv[2]
    #run_id = '20210929_seq1_normal_mouseBrain'
    ref_file = './plex_map_filtered_108plex_10base_barcode.csv'
    dest_directory = os.path.join(BASE_DEST_DIRECTORY,f'{run_id}_processed')
    stc_directory = os.path.join(dest_directory,'stitched')
    read_directory = os.path.join(dest_directory,'readout')
    try:
        os.mkdir(read_directory)
    except FileExistsError:
        pass
    
    extract_raw_sequence(stc_directory,read_directory,THRESHOLDS)

    checked_df = check_sequence(os.path.join(read_directory,RAW_SEQ_NAME),ref_file)
    checked_df.to_csv(os.path.join(read_directory,CHECKED_NAME),index=False)
    map_df = map_barcode(os.path.join(read_directory,CHECKED_NAME),ref_file)
    map_df = unstack_plex(map_df)
    map_df.to_csv(os.path.join(read_directory,'mapped_genes.csv'),index=False)

def extract_test():
    RUN_ID = '20211119_36plex_1'
    cyc_chn = 'cyc_12_cy5'
    dest_directory = os.path.join(BASE_DEST_DIRECTORY,f'{RUN_ID}_processed')
    stc_directory = os.path.join(dest_directory,'stitched')
    raw_im = imread(os.path.join(stc_directory,f'{cyc_chn}.tif'))[15000:20000,15000:20000]
    im = tophat_spots(raw_im)
    coordinates,meta_data = extract_coordinates(im,snr=5,quantile=0.2)
    output_im = np.zeros((5000,5000),dtype=np.uint16)
    output_im[coordinates[:,0],coordinates[:,1]] = 65535
    imsave(os.path.join(stc_directory,f'{cyc_chn}_spots.tif'),output_im)
    imsave(os.path.join(stc_directory,f'{cyc_chn}_roi.tif'),raw_im)
    print(meta_data)

def normalize_sample():
    RUN_ID = '20211119_36plex_1'
    dest_directory = os.path.join(BASE_DEST_DIRECTORY,f'{RUN_ID}_processed')
    stc_directory = os.path.join(dest_directory,'stitched')
    read_directory = os.path.join(dest_directory,'readout')
    df = pd.read_csv(os.path.join(read_directory,'intensity_sample.csv'))
    #print(df.iloc[:,2:].mean())
    #print(df.iloc[:,2:].mean(axis=1))
    df.iloc[:,2:32:2] = df.iloc[:,2:32:2].to_numpy() - df.iloc[:,2:32:2].mean(axis=1).to_numpy().astype(int)[:,None]
    df.iloc[:,3:32:2] = df.iloc[:,3:32:2].to_numpy() - df.iloc[:,3:32:2].mean(axis=1).to_numpy().astype(int)[:,None]
    df[df<0] = 0.001
    df.iloc[:,2:] = np.log10(df.iloc[:,2:])
    df[df<2] = 0
    df.to_csv(os.path.join(read_directory,'intensity_sample_centered.csv'),index=False)

if __name__ == "__main__":
    # check filter_by_count before running
    main('20230330_20x_AD_4')