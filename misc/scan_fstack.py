import time
import os
import re
import sys
from glob import glob
import pandas as pd
import matlab.engine
from tqdm import tqdm

BASE_DIRECTORY = r'D:\FISH_images'
BASE_DEST_DIRECTORY = r'E:\FISH_images_processed'
CHANNELS = ['cy3','cy5','FAM','DAPI']
INTERVAL = 600 # in seconds
TIMEOUT = 10800

def try_mkdir(d):
    """Try to make a new directory d. Passes if it already exists."""
    try:
        os.makedirs(d)
    except FileExistsError:
        print(f'{d} exists.')
        pass

def imlist_to_df(imlist):
    """Given a list of properly named .tif files, return a dataframe with organized cycle, tile, channel and depth information."""
    imlist_2d = [s.strip('.tif').split('-') for s in imlist]
    df = pd.DataFrame(imlist_2d, columns=['Cycle','Tile','Channel','Z'])
    df = df.loc[:,['Cycle','Tile','Z']]
    for column in ['Cycle','Tile','Z']:
        df[column] = df[column].apply(lambda x: int(x[1:]))
    return df

def get_stack_num(path):
    """Given a directory, return max z-axis index plus one (which is the image count of a stack)."""
    sample = [f for f in os.listdir(path) if f.endswith('.tif')][0]
    prefix = sample.split('-Z')[0]
    stack_num = max([int(f.strip('.tif').split('-Z')[1]) for f in os.listdir(path) if f.startswith(prefix)]) + 1
    return stack_num

def get_cycles(d,even_less=False,zero=False):
    """Given a directory, return full path of folders named cyc_*.
    
    Keyword arguments:
    even_less -- only odd cycles will be included (default False)
    zero -- include cycle 0 (default False)
    """
    if not even_less:
        cycles = glob(os.path.join(d,'cyc_*'))
    else:
        cycles = glob(os.path.join(d,'cyc_*[13579]'))
        if zero:
            cycles += glob(os.path.join(d,'cyc_0'))
    return cycles

def stack_cycle(in_directory,out_directory,cycle):
    """Full all-in-focus image generation pipeline, given Run ID."""
    eng = matlab.engine.start_matlab()
    stack_num = get_stack_num(os.path.join(in_directory,cycle))
    chn_info = {}
    for chn in CHANNELS:
        if glob(os.path.join(in_directory,cycle,f'*-{chn}-*')):
            chn_info[chn] = imlist_to_df([f for f in os.listdir(os.path.join(in_directory,cycle)) if chn in f])
            cyc = chn_info[chn].loc[0,'Cycle']
            cyc_chn_directory = os.path.join(out_directory,f'cyc_{cyc}_{chn}')
            try:
                os.mkdir(cyc_chn_directory)
            except FileExistsError:
                print(f'Cycle {cyc}, Channel {chn} exists.')
                continue
            captured_tiles = list(chn_info[chn]['Tile'].unique())
            for tile in tqdm(captured_tiles,desc=f'Cycle {cyc}, Channel {chn}'):
                z_stack_names = [os.path.join(in_directory,cycle,f'C{cyc:03d}-T{tile:04d}-{chn}-Z{z:03d}.tif') for z in range(stack_num)]
                eng.fstack_write(z_stack_names,os.path.join(out_directory,f'cyc_{cyc}_{chn}',f'FocalStack_{tile:03d}.tif'),nargout=0)
    eng.quit()

def main():
    wait = 0
    run_id = sys.argv[1]
    src_directory = os.path.join(BASE_DIRECTORY,run_id)
    dest_directory = os.path.join(BASE_DEST_DIRECTORY,f'{run_id}_processed','focal_stacked')
    try_mkdir(dest_directory)
    
    while True:
        cycles = [f for f in os.listdir(src_directory) if re.match(r'^cyc_([0-9]+)$',f) and f!='cyc_0']
        print(cycles)
        for cyc in cycles:
            if glob(os.path.join(src_directory,cyc,'finish.txt')) and not glob(os.path.join(dest_directory,f'{cyc}*')):
                wait = 0
                print(f'Found cycle to stack. Directory {cyc}')
                stack_cycle(src_directory,dest_directory,cyc)
        else:
            print(f'Wait for {INTERVAL} seconds...')
            time.sleep(INTERVAL)
            wait += INTERVAL
        if wait > TIMEOUT:
            print('Wait timeout. Session ended.')
            break

if __name__ == "__main__":
    main()
