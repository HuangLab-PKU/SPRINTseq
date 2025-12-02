import re
from pathlib import Path
from tkinter import image_names
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread
from skimage.io import imsave
import matlab.engine


def patch_tiles(in_dir, tile_num):
    """Patch missing tiles in a directory.

    Parameters
    ----------
    in_dir : str
        Path to image tiles.
    tile_num : int
        Number of tiles total.

    Returns
    -------
    None.

    """
    im_names = [f'FocalStack_{t:03d}.tif' for t in range(1,tile_num+1)]
    ref_im = imread(list(Path(in_dir).glob('*.tif'))[0])
    temp_im = np.zeros(ref_im.shape, dtype=np.uint16)
    for im_name in im_names:
        im_path = Path(in_dir)/im_name
        if not im_path.is_file():
            imsave(str(im_path), temp_im, check_contrast=False)


def template_stitch(in_dir, out_dir, tile_x, tile_y):
    """Stitch a series of images in a directory.

    Parameters
    ----------
    in_dir : str
        Path to source images.
    out_dir : str
        Path to stitched images.
    tile_x : int
        Number of tiles in horizontal direction.
    tile_y : int
        Number of tiles in vertical direction.
    
    Returns
    -------
    None.

    """
    eng = matlab.engine.start_matlab()
    mist_path = (Path(__file__).parent)/'MIST-MATLAB'
    mist_src_path = mist_path/'src'/'subfunctions'
    eng.addpath(str(mist_path))
    eng.addpath(str(mist_src_path))
    eng.stitch_silent(str(in_dir), str(out_dir), tile_x, tile_y, nargout=0)
    eng.quit()


def read_meta(in_dir):
    """Returns metadata from global position file.

    Parameters
    ----------
    in_dir : str
        Path to stitched folder.

    Returns
    -------
    meta_df : pandas.DataFrame
        DataFrame with image tile metadata.

    """
    meta_path = list(Path(in_dir).glob('*global*.txt'))[0]
    raw = ['\n'.join(line.strip(';').split('; '))
           for line in open(meta_path).read().strip('\n').split('\n')]
    cols, vals = zip(*[line.split(': ')
                     for line in '\n'.join(raw).split('\n')])
    series = pd.Series(vals, cols)
    series.index = [series.groupby(level=0).cumcount(), series.index]
    meta_df = series.unstack()
    return meta_df


def border_calc(x, bleed, tile_width, border):
    """Returns the border of the image (1D).

    Parameters
    ----------
    x : int
        Assumed tile start position.
    bleed : int
        Number of pixels to add to the border.
    tile_width : int
        Width of the tile.
    border : int
        Maximum tolerated value (end position).

    Returns
    -------
    x_start : int
        Start position within the tile.
    x_end : int
        End position within the tile.

    """
    if x < 0:
        x_start = bleed - x
    else:
        x_start = bleed
    if x + tile_width > border:
        x_end = border - x - bleed
    else:
        x_end = tile_width - bleed
    return x_start, x_end


def stitch_manual(in_dir, out_dir, shift_df, cyc=None, bleed=20):
    sample_im = imread(list(in_dir.glob('*.tif'))[0])
    tile_width = sample_im.shape[1]
    meta_df = read_meta(out_dir)
    pattern = r'\((\d+)\, *(\d+)\)'
    meta_df['match'] = meta_df['position'].apply(
        lambda x: re.match(pattern, x))
    meta_df['y'] = meta_df['match'].apply(lambda x: int(x.group(2)))
    meta_df['x'] = meta_df['match'].apply(lambda x: int(x.group(1)))
    height = max(meta_df['y']) + tile_width
    width = max(meta_df['x']) + tile_width
    if cyc is None:
        cyc = int(in_dir.name.split('_')[1])
    rgs_name = in_dir.name
    canvas = np.zeros((height, width), dtype=np.uint16)
    for _, row in meta_df.iterrows():
        if not (Path(in_dir)/row.file).is_file():
            continue
        temp_im = imread(Path(in_dir)/row.file)
        shift = shift_df.loc[cyc, row.file]
        shift = ' '.join(shift.split())
        y_shift, x_shift = shift.split(' ')
        y, x = int(y_shift) + row.y, int(x_shift) + row.x
        x_start, x_end = border_calc(x, bleed, tile_width, width)
        y_start, y_end = border_calc(y, bleed, tile_width, height)
        canvas[max(y, 0)+bleed:min(y+tile_width, height)-bleed, max(x, 0) +
                bleed:min(x+tile_width, width)-bleed] = temp_im[y_start:y_end, x_start:x_end]
    imsave(str(Path(out_dir)/f'{rgs_name}.tif'), canvas, check_contrast=False)


def stitch_offset(in_dir, out_dir, shift_df, bleed=20, ref_cyc=1, auto_skip=True):
    """Stitches images with given global positions and offsets.

    Parameters
    ----------
    in_dir : str
        Path to registered images.
    out_dir : str
        Path to stitched images.
    shift_df : pandas.DataFrame
        DataFrame with integer offsets.
    bleed : int
        Number of pixels to add to the border.
    ref_cyc : int
        Reference cycle number for stitching.

    Returns
    -------
    None.

    """
    rgs_list = list(Path(in_dir).glob('cyc_[0-9]*_*'))
    sample_dir = rgs_list[0]
    sample_im = imread(list(sample_dir.glob('*.tif'))[0])
    sample_dtype = type(sample_im[0,0])
    tile_width = sample_im.shape[1]
    meta_df = read_meta(out_dir)
    pattern = r'\((\d+)\, *(\d+)\)'
    meta_df['match'] = meta_df['position'].apply(
        lambda x: re.match(pattern, x))
    meta_df['y'] = meta_df['match'].apply(lambda x: int(x.group(2)))
    meta_df['x'] = meta_df['match'].apply(lambda x: int(x.group(1)))
    height = max(meta_df['y']) + tile_width
    width = max(meta_df['x']) + tile_width
    for rgs_dir in tqdm(rgs_list, desc='Stitching'):
        cyc = int(rgs_dir.name.split('_')[1])
        rgs_name = rgs_dir.name
        if (Path(out_dir)/f'{rgs_name}.tif').is_file() and auto_skip:
            continue
        # Skip cycles that are not in shift_df (they may be processed manually later)
        if cyc != ref_cyc and cyc not in shift_df.index:
            tqdm.write(f'Skipping {rgs_name}: cycle {cyc} not found in offset_df. It may be processed manually.')
            continue
        canvas = np.zeros((height, width), dtype=sample_dtype)
        for _, row in meta_df.iterrows():
            if not (Path(rgs_dir)/row.file).is_file():
                continue
            temp_im = imread(Path(rgs_dir)/row.file)
            if cyc == ref_cyc:
                x, y = row.x, row.y
            else:
                shift = shift_df.loc[cyc, row.file]
                shift = ' '.join(shift.split())
                y_shift, x_shift = shift.split(' ')
                y, x = int(y_shift) + row.y, int(x_shift) + row.x
            x_start, x_end = border_calc(x, bleed, tile_width, width)
            y_start, y_end = border_calc(y, bleed, tile_width, height)
            canvas[max(y, 0)+bleed:min(y+tile_width, height)-bleed, max(x, 0) +
                   bleed:min(x+tile_width, width)-bleed] = temp_im[y_start:y_end, x_start:x_end]
        imsave(str(Path(out_dir)/f'{rgs_name}.tif'), canvas, check_contrast=False)


def main():
    in_dir = '/Users/Leon/Downloads/stitched'
    print(read_meta(in_dir).head())


if __name__ == '__main__':
    main()
