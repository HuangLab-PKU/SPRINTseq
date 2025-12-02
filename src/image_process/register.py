from .utils.os_utils import try_mkdir
import re
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.io import imsave
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift


def get_shift(ref, img):
    """Get the shift between two images.

    Parameters
    ----------
    ref : ndarray
        Reference image.
    img : ndarray
        Image to register.

    Returns
    -------
    shift : tuple
        (y, x) shift.

    """
    # For scikit-image version compatibility
    # - In newer versions (e.g., >=0.24.0), we explicitly pass normalization=None to disable phase normalization,
    #   making the behavior closer to older versions like 0.18.1.
    # - In older versions (such as 0.18.1), phase_cross_correlation does not support the normalization argument,
    #   so passing it will raise a TypeError. In that case, we fall back to calling without this argument.
    try:
        # Newer API (supports normalization argument)
        shift, _, _ = phase_cross_correlation(ref, img, upsample_factor=100, normalization=None)
    except TypeError:
        # Older API (does not support normalization argument)
        shift, _, _ = phase_cross_correlation(ref, img, upsample_factor=100)
    return shift


def register(img, shift):
    """Register an image to a reference image.

    Parameters
    ----------
    img : ndarray
        Image to register.
    shift : tuple
        (y, x) shift.

    Returns
    -------
    out_img : ndarray
        Registered image.

    """
    registered = fourier_shift(np.fft.fftn(img), shift)
    registered = np.fft.ifftn(registered)
    registered_real = np.clip(registered.real, 0, 65535)
    out_img = registered_real.astype(np.uint16)
    return out_img


def register_manual(ref_dir, src_dir, dest_dir, im_names=None):
    """Register all images in a directory.
    
    Parameters
    ----------
    ref_dir : str
        Directory with the reference images.
    src_dir : str
        Directory with the images to register.
    dest_dir : str
        Output directory.
    im_names : list
        Names of the images to register.

    Returns
    -------
    None.
    
    """
    ref_dir = Path(ref_dir)
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)
    if im_names is None:
        im_names = [x.name for x in list(ref_dir.glob('*.tif'))]
    for im_name in tqdm(im_names, desc='Registering'):
        ref_im = imread(ref_dir/im_name)
        src_im = imread(src_dir/im_name)
        shift = get_shift(ref_im, src_im)
        out_im = register(src_im, shift)
        imsave(dest_dir/im_name, out_im, check_contrast=False)


def register_meta(in_dir, out_dir, chns, names, ref_cyc=1, ref_chn='cy3'):
    """Register all images in a directory.

    Parameters
    ----------
    in_dir : str
        Input directory.
    out_dir : str
        Output directory.
    chns : list
        Channels to register.
    names : list
        Names of the images to register.
    ref_cyc : int
        Cycle to use as reference.
    ref_chn : str
        Channel to use as reference.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame of the integer offsets for stitching.

    """
    try_mkdir(out_dir)
    pattern = r'^cyc_\d+_\w+'
    alt_chns = [c for c in chns if c != ref_chn]
    cyc_chn_list = Path(in_dir).glob('cyc_*_*')
    cyc_chn_list = [c for c in cyc_chn_list if re.match(pattern, c.name)]
    src_list = [c for c in cyc_chn_list if c.name.split('_')[
        1] == str(ref_cyc)]
    for d in tqdm(src_list, desc='Copying reference'):
        if not (Path(out_dir)/d.name).is_dir():
            shutil.copytree(d, Path(out_dir)/d.name)
    df = pd.DataFrame()
    ref_list = [c for c in cyc_chn_list if c.name.split('_')[2] == ref_chn]
    ref_dir = Path(in_dir)/f'cyc_{ref_cyc}_{ref_chn}'
    ref_list.remove(ref_dir)
    for name in tqdm(names, desc='Registering'):
        offsets = []
        ref_im = imread(ref_dir/name)
        for cyc_chn in ref_list:
            cyc = cyc_chn.name.split('_')[1]
            src_im = imread(cyc_chn/name)
            dest_dir = Path(out_dir)/cyc_chn.name
            dest_dir.mkdir(exist_ok=True)
            try:
                shift = get_shift(ref_im, src_im)
            except AttributeError:
                offsets.append('0 0')
                continue
            shift_res = shift - np.round(shift)
            offsets.append(np.array2string(np.round(shift).astype(int)).strip('[ ]'))
            out_im = register(src_im, shift_res)
            imsave(dest_dir/name, out_im, check_contrast=False)
            for chn in alt_chns:
                cyc_chn_alt = Path(in_dir)/f'cyc_{cyc}_{chn}'
                if cyc_chn_alt in cyc_chn_list:
                    dest_dir_alt = Path(out_dir)/cyc_chn_alt.name
                    dest_dir_alt.mkdir(exist_ok=True)
                    src_im = imread(cyc_chn_alt/name)
                    out_im = register(src_im, shift_res)
                    imsave(dest_dir_alt/name, out_im, check_contrast=False)
        df[name] = offsets
    df.index = [s.name.split('_')[1] for s in ref_list]
    return df


def register_shared_offset(ref_dir, src_base_dir, dest_base_dir, target_cyc, channels, offset_channel=None, im_names=None):
    """Register multiple channels using the same offset calculated from a reference channel.
    
    This function is useful when multiple channels (e.g., DAPI and FAM) are captured 
    simultaneously and should share the same offset. The offset is calculated using 
    a specified channel (or the first channel by default) against a reference, 
    then applied to all specified channels.
    
    Parameters
    ----------
    ref_dir : str or Path
        Directory with the reference images (e.g., cyc_10_DAPI).
    src_base_dir : str or Path
        Base directory containing source images (e.g., sdc_dir).
    dest_base_dir : str or Path
        Base directory for output registered images (e.g., rgs_dir).
    target_cyc : int
        Target cycle number (e.g., 12).
    channels : list
        List of channel names to register (e.g., ['DAPI', 'FAM']).
        The offset will be applied to all channels in this list.
    offset_channel : str, optional
        Channel name to use for calculating offset (e.g., 'DAPI' or 'FAM').
        If None, uses the first channel in the channels list.
        This channel must be in the channels list.
    im_names : list, optional
        Names of the images to register. If None, uses all .tif files in ref_dir.
    
    Returns
    -------
    offset_df : pandas.DataFrame
        DataFrame with integer offsets for stitching. Index is cycle number, 
        columns are image names, values are offset strings like "y x".
    
    """
    ref_dir = Path(ref_dir)
    src_base_dir = Path(src_base_dir)
    dest_base_dir = Path(dest_base_dir)
    
    # Get image names if not provided
    if im_names is None:
        im_names = [x.name for x in list(ref_dir.glob('*.tif'))]
    
    # Determine which channel to use for offset calculation
    if offset_channel is None:
        offset_channel = channels[0]
    elif offset_channel not in channels:
        raise ValueError(f"offset_channel '{offset_channel}' must be in the channels list: {channels}")
    src_offset_dir = src_base_dir / f'cyc_{target_cyc}_{offset_channel}'
    
    # Calculate offsets for each image
    offsets_dict = {}
    shift_residuals = {}  # Store residual shifts for registration
    
    for im_name in tqdm(im_names, desc=f'Calculating offsets for cyc_{target_cyc}'):
        # Read reference and source images for offset calculation
        ref_im = imread(ref_dir / im_name)
        src_im = imread(src_offset_dir / im_name)
        
        # Calculate shift
        shift = get_shift(ref_im, src_im)
        
        # Store integer offset for stitching (as string "y x")
        integer_offset = np.round(shift).astype(int)
        offsets_dict[im_name] = f"{integer_offset[0]} {integer_offset[1]}"
        
        # Store residual shift (non-integer part) for registration
        shift_residuals[im_name] = shift - np.round(shift)
    
    # Create offset DataFrame
    offset_df = pd.DataFrame([offsets_dict], index=[str(target_cyc)])
    
    # Apply the same offset to all channels
    for chn in tqdm(channels, desc=f'Registering channels for cyc_{target_cyc}'):
        src_chn_dir = src_base_dir / f'cyc_{target_cyc}_{chn}'
        dest_chn_dir = dest_base_dir / f'cyc_{target_cyc}_{chn}'
        dest_chn_dir.mkdir(parents=True, exist_ok=True)
        
        for im_name in tqdm(im_names, desc=f'  Registering {chn}', leave=False):
            src_im = imread(src_chn_dir / im_name)
            # Apply the residual shift (non-integer part) for registration
            shift_res = shift_residuals[im_name]
            out_im = register(src_im, shift_res)
            imsave(dest_chn_dir / im_name, out_im, check_contrast=False)
    
    return offset_df


def main():
    pass


if __name__ == "__main__":
    main()
