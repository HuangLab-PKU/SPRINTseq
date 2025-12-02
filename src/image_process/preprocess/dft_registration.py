import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.io import imsave
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
import cv2
import shutil
from tqdm import tqdm
from glob import glob
from pathlib import Path

def registered(image,offset_image):
    shift,_,_ = phase_cross_correlation(image,offset_image,upsample_factor=100)
    registered_image = fourier_shift(np.fft.fftn(offset_image),shift)
    registered_image = np.fft.ifftn(registered_image)
    return np.uint16(registered_image.real)

def get_shift(image,offset_image):
    shift,_,_ = phase_cross_correlation(image,offset_image,upsample_factor=100)
    return shift

def register_with_shift(offset_image,shift):
    registered_image = fourier_shift(np.fft.fftn(offset_image),shift)
    registered_image = np.fft.ifftn(registered_image)
    return np.uint16(registered_image.real)

def register_pair(input_directory,registered_directory,cyc_chn_ref,cyc_chn_offset,img_names):
    try:
        os.mkdir(os.path.join(registered_directory,cyc_chn_offset))
    except FileExistsError:
        pass
    for name in tqdm(img_names):
        ref_image = imread(os.path.join(input_directory,cyc_chn_ref,name))
        offset_image = imread(os.path.join(input_directory,cyc_chn_offset,name))
        shift = get_shift(ref_image,offset_image)
        imsave(os.path.join(registered_directory,cyc_chn_offset,name),register_with_shift(offset_image,shift),check_contrast=False)

def register_channel(input_directory,registered_directory,ref_cycle,channel,img_names):
    ref_cyc_chn = os.path.join(registered_directory,f'cyc_{ref_cycle}_{channel}')
    if not glob(os.path.join(ref_cyc_chn,img_names[0])):
        try:
            os.mkdir(ref_cyc_chn)
        except FileExistsError:
            pass
        for img in img_names:
            shutil.copy(os.path.join(input_directory,f'cyc_{ref_cycle}_{channel}',img),os.path.join(ref_cyc_chn,img))
    cyc_chn_list = [f for f in glob(os.path.join(input_directory,f'cyc_*_{channel}')) if f!=ref_cyc_chn]
    cyc_chn_rgs_list = [f.replace(input_directory,registered_directory) for f in cyc_chn_list]
    for d in cyc_chn_rgs_list:
        try:
            os.mkdir(d)
        except FileExistsError:
            pass
    for name in tqdm(img_names):
        ref_image = imread(os.path.join(ref_cyc_chn,name))
        for d,d_rgs in zip(cyc_chn_list,cyc_chn_rgs_list):
            rgs_image = registered(ref_image,imread(os.path.join(d,name)))
            imsave(os.path.join(d_rgs,name),rgs_image,check_contrast=False)

def interchannel_correction(input_directory,registered_directory,cycle,ref_channel,img_names):
    ref_cyc_chn = os.path.join(registered_directory,f'cyc_{cycle}_{ref_channel}')
    if not glob(os.path.join(ref_cyc_chn,img_names[0])):
        try:
            os.mkdir(ref_cyc_chn)
        except FileExistsError:
            pass
        for img in img_names:
            shutil.copy(os.path.join(input_directory,f'cyc_{cycle}_{ref_channel}',img),os.path.join(ref_cyc_chn,img))
    cyc_chn_list = [f for f in glob(os.path.join(input_directory,f'cyc_{cycle}_*')) if f!=ref_cyc_chn]
    cyc_chn_rgs_list = [f.replace(input_directory,registered_directory) for f in cyc_chn_list]
    for d in cyc_chn_rgs_list:
        try:
            os.mkdir(d)
        except FileExistsError:
            pass
    for name in tqdm(img_names):
        ref_image = imread(os.path.join(ref_cyc_chn,name))
        for d,d_rgs in zip(cyc_chn_list,cyc_chn_rgs_list):
            rgs_image = registered(ref_image,imread(os.path.join(d,name)))
            imsave(os.path.join(d_rgs,name),rgs_image,check_contrast=False)



def register_with_ref(input_directory,registered_directory,channels,img_names,ref_cycle=1,ref_channel='DAPI'):
    alter_channels = [c for c in channels if c!=ref_channel]
    cyc_chn_list = [f for f in os.listdir(input_directory) if f.startswith('cyc_')]
    ref_channels_src = glob(os.path.join(input_directory,f'cyc_{ref_cycle}_*'))
    for cyc_chn in ref_channels_src:
        cyc_chn_cp = cyc_chn.replace(input_directory,registered_directory)
        try:
            os.mkdir(cyc_chn_cp)
        except FileExistsError:
            continue
        for img in os.listdir(cyc_chn):
            shutil.copy(os.path.join(cyc_chn,img),os.path.join(cyc_chn_cp,img))
    ref_cyc_chn = os.path.join(registered_directory,f'cyc_{ref_cycle}_{ref_channel}')
    ref_cyc_list = [f for f in cyc_chn_list if ref_channel in f]
    ref_cyc_list.remove(f'cyc_{ref_cycle}_{ref_channel}')
    for name in tqdm(img_names):
        ref_image = cv2.imread(os.path.join(ref_cyc_chn,name),-cv2.IMREAD_ANYDEPTH)
        for cyc_chn in ref_cyc_list:
            temp_source_path = os.path.join(input_directory,cyc_chn)
            temp_dest_path = os.path.join(registered_directory,cyc_chn)
            try:
                os.makedirs(temp_dest_path)
            except FileExistsError:
                pass
            if len(os.listdir(temp_dest_path))>=len(img_names):
                pass
            offset_image = cv2.imread(os.path.join(temp_source_path,name),-cv2.IMREAD_ANYDEPTH)
            try:
                shift = get_shift(ref_image,offset_image)
            except AttributeError:
                continue
            imsave(os.path.join(temp_dest_path,name),register_with_shift(offset_image,shift),check_contrast=False)
            for chn in alter_channels:
                cyc_alter_chn = cyc_chn.replace(ref_channel,chn)
                if cyc_alter_chn in cyc_chn_list:
                    temp_source_path = os.path.join(input_directory,cyc_alter_chn)
                    temp_dest_path = os.path.join(registered_directory,cyc_alter_chn)
                    try:
                        os.makedirs(temp_dest_path)
                    except FileExistsError:
                        pass
                    offset_image = cv2.imread(os.path.join(temp_source_path,name),-cv2.IMREAD_ANYDEPTH)
                    imsave(os.path.join(temp_dest_path,name),register_with_shift(offset_image,shift),check_contrast=False)

def register_meta(input_directory,registered_directory,channels,img_names,ref_cycle=1,ref_channel='DAPI'):
    alter_channels = [c for c in channels if c!=ref_channel]
    cyc_chn_list = [f for f in os.listdir(input_directory) if f.startswith('cyc_')]
    ref_channels_src = glob(os.path.join(input_directory,f'cyc_{ref_cycle}_*'))
    for cyc_chn in ref_channels_src:
        cyc_chn_cp = cyc_chn.replace(input_directory,registered_directory)
        try:
            os.mkdir(cyc_chn_cp)
        except FileExistsError:
            continue
        for img in os.listdir(cyc_chn):
            shutil.copy(os.path.join(cyc_chn,img),os.path.join(cyc_chn_cp,img))
    ref_cyc_chn = os.path.join(registered_directory,f'cyc_{ref_cycle}_{ref_channel}')
    ref_cyc_list = [f for f in cyc_chn_list if ref_channel in f]
    ref_cyc_list.remove(f'cyc_{ref_cycle}_{ref_channel}')
    df = pd.DataFrame()
    for name in tqdm(img_names):
        offsets = []
        ref_image = cv2.imread(os.path.join(ref_cyc_chn,name),-cv2.IMREAD_ANYDEPTH)
        for cyc_chn in ref_cyc_list:
            temp_source_path = os.path.join(input_directory,cyc_chn)
            temp_dest_path = os.path.join(registered_directory,cyc_chn)
            try:
                os.makedirs(temp_dest_path)
            except FileExistsError:
                pass
            if len(os.listdir(temp_dest_path))>=len(img_names):
                continue
            offset_image = cv2.imread(os.path.join(temp_source_path,name),-cv2.IMREAD_ANYDEPTH)
            try:
                shift = get_shift(ref_image,offset_image)
            except AttributeError:
                offsets.append('0 0')
                continue
            shift_remainder = shift - np.round(shift)
            offsets.append(np.array2string(np.round(shift).astype(int)).strip('[ ]'))
            imsave(os.path.join(temp_dest_path,name),register_with_shift(offset_image,shift_remainder),check_contrast=False)
            for chn in alter_channels:
                cyc_alter_chn = cyc_chn.replace(ref_channel,chn)
                if cyc_alter_chn in cyc_chn_list:
                    temp_source_path = os.path.join(input_directory,cyc_alter_chn)
                    temp_dest_path = os.path.join(registered_directory,cyc_alter_chn)
                    try:
                        os.makedirs(temp_dest_path)
                    except FileExistsError:
                        pass
                    try:
                        offset_image = imread(os.path.join(temp_source_path,name))
                    except FileNotFoundError:
                        continue
                    if (Path(temp_dest_path)/name).is_file():
                        continue
                    else:
                        imsave(os.path.join(temp_dest_path,name),register_with_shift(offset_image,shift_remainder),check_contrast=False)
        df[name] = offsets
    df.index = [s.split('_')[1] for s in ref_cyc_list]
    return df

if __name__ == "__main__":
    #print([f'{i:03d}' for i in range(99)])
    register_pair(r'\\10.10.10.1\NAS Processed Images\20211104_Mask_control_processed\background_corrected',r'\\10.10.10.1\NAS Processed Images\20211104_Mask_control_processed\registered','cyc_10_DAPI','cyc_11_DAPI',[f'FocalStack_{i+5:03d}.tif' for i in range(95)])


