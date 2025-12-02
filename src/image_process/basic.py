"""
BaSiCPy background correction module

Uses BaSiCPy (fast JAX-based background correction algorithm) to replace CIDRE for illumination correction.
BaSiCPy is pure Python implementation, very fast, no MATLAB engine required.
"""

import re
import os
import shutil
import warnings
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import numpy as np
from skimage.io import imread, imsave
import random
from basicpy import BaSiC


def basic_walk(in_dir, out_dir, fmt='tif', 
               get_darkfield=True, smoothness_flatfield=1.0,
               sample_size=None, batch_size=10):
    """Correct images using BaSiCPy, grouped by channel.
    
    This function groups all cycles of the same channel together, builds a single
    BaSiC model for each channel, then applies correction while maintaining the
    original directory structure (cyc_X_channel).
    
    BaSiCPy handles bright tiles more robustly than CIDRE, no additional filtering needed.

    Parameters
    ----------
    in_dir : str
        Input directory path containing cyc_*_* subdirectories
    out_dir : str
        Output directory path
    fmt : str, optional
        Image format (default='tif')
    get_darkfield : bool, optional
        Whether to compute dark field (default=True)
        If True, corrects both illumination non-uniformity and dark current
    smoothness_flatfield : float, optional
        Flat field smoothness parameter (default=1.0)
        Higher values produce smoother flat field, suitable for slowly varying illumination
        For cases with strong sample background, can increase to 2.0-3.0
    sample_size : int, optional
        Number of images to use for flat field computation (default=None, auto-select based on total)
        If specified, randomly selects this many images to compute flat field, then applies to all
        For large datasets (1000+ images), 50-100 images is usually sufficient
        If None, will auto-select: min(100, total_images * 0.1)
    batch_size : int, optional
        Number of images to process in each batch during transform (default=10)
        Larger batch_size uses more memory but may be faster
        For very large images or limited memory, reduce to 5 or even 1

    Returns
    -------
    None

    Notes
    -----
    - Memory-efficient two-stage processing:
      1. Fit stage: Randomly samples images to compute flat field (uses minimal memory)
      2. Transform stage: Processes images in batches to apply correction (streaming)
    - BaSiCPy automatically downsamples images to compute flat field (since flat field is low-frequency),
      then applies back to original resolution, making it very fast
    - BaSiCPy handles bright tiles more robustly, no filtering needed like CIDRE
    - For large datasets (1000+ images), 50-100 sample images is usually sufficient
    - BaSiCPy uses JAX, automatically uses GPU if available
    - Output directory structure matches input (cyc_X_channel)

    Examples
    --------
    >>> # Use default parameters
    >>> basic_walk('input_dir', 'output_dir')
    
    >>> # Use smoother flat field (suitable for strong sample background)
    >>> basic_walk('input_dir', 'output_dir', smoothness_flatfield=2.0)
    
    >>> # Randomly sample 100 images for model computation (speed up large datasets)
    >>> basic_walk('input_dir', 'output_dir', sample_size=100)
    
    >>> # Disable dark field correction (only correct illumination)
    >>> basic_walk('input_dir', 'output_dir', get_darkfield=False)

    """
    
    # Collect all cyc_*_* directories
    p = Path(in_dir).glob('cyc_[0-9]*_*')
    pattern = r'^cyc_\d+_(\w+)$'  # Extract channel name
    sub_dirs = [x for x in p if x.is_dir()]
    sub_dirs = [x for x in sub_dirs if re.match(pattern, x.name)]
    
    # Group by channel
    channel_groups = defaultdict(list)
    for sub_dir in sub_dirs:
        match = re.match(pattern, sub_dir.name)
        if match:
            channel = match.group(1)
            channel_groups[channel].append(sub_dir)
    
    if not channel_groups:
        print("Warning: No directories matching cyc_*_* pattern found")
        return
    
    # Ensure output directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Process each channel
    channel_pbar = tqdm(channel_groups.items(), desc='BaSiC correcting by channel', position=0)
    for channel, dirs in channel_pbar:
        # Check if all directories are already processed
        all_processed = True
        total_src = 0
        total_dest = 0
        
        for sub_dir in dirs:
            out_sub_dir = Path(out_dir) / sub_dir.name
            out_sub_dir.mkdir(parents=True, exist_ok=True)
            src_cnt = len(list(sub_dir.glob(f'*.{fmt}')))
            dest_cnt = len(list(out_sub_dir.glob(f'*.{fmt}')))
            total_src += src_cnt
            total_dest += dest_cnt
            if src_cnt != dest_cnt:
                all_processed = False
        
        # Skip if all images are already processed
        if all_processed and total_src > 0:
            continue
        
        # Collect all image paths
        all_image_paths = []  # Store (sub_dir, img_path) tuples
        
        for sub_dir in dirs:
            images = list(sub_dir.glob(f'*.{fmt}'))
            for img_path in images:
                all_image_paths.append((sub_dir, img_path))
        
        if not all_image_paths:
            tqdm.write(f"  Warning: {channel} channel has no images to process")
            continue
        
        total_images = len(all_image_paths)
        tqdm.write(f"  {channel} channel: {total_images} images to process")
        
        # Stage 1: Fit - Randomly sample images to compute flat field
        # Auto-select sample_size if not specified
        if sample_size is None:
            sample_size = min(100, max(50, int(total_images * 0.1)))
        
        if total_images > sample_size:
            tqdm.write(f"  Stage 1: Randomly sampling {sample_size} images for flat field computation...")
            fit_indices = random.sample(range(total_images), sample_size)
            fit_paths = [all_image_paths[i] for i in fit_indices]
        else:
            tqdm.write(f"  Stage 1: Using all {total_images} images for flat field computation...")
            fit_paths = all_image_paths
        
        # Load only sampled images for fitting
        fit_images = []
        for sub_dir, img_path in fit_paths:
            try:
                img = imread(str(img_path))
                if len(img.shape) != 2:
                    tqdm.write(f"  Warning: {img_path.name} is not a 2D image, skipping")
                    continue
                fit_images.append(img)
            except Exception as e:
                tqdm.write(f"  Warning: Failed to read {img_path.name}: {e}")
                continue
        
        if not fit_images:
            tqdm.write(f"  Error: {channel} channel has no valid images for fitting")
            continue
        
        # Convert to numpy array and compute flat field
        fit_images = np.array(fit_images)
        tqdm.write(f"  Fitting on {len(fit_images)} images (shape: {fit_images.shape})...")
        
        try:
            basic = BaSiC(get_darkfield=get_darkfield, smoothness_flatfield=smoothness_flatfield)
            # Suppress convergence warning (this is usually not critical)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*Reweighting did not converge.*')
                warnings.filterwarnings('ignore', category=UserWarning)
                basic.fit(fit_images)
            tqdm.write(f"  Flat field computation completed")
        except Exception as e:
            tqdm.write(f"  Error: Failed to compute flat field: {e}")
            continue
        
        # Release fit_images memory
        del fit_images
        
        # Stage 2: Transform - Process images in batches
        tqdm.write(f"  Stage 2: Applying correction to {total_images} images in batches of {batch_size}...")
        
        # Process images in batches to save memory
        num_batches = (total_images + batch_size - 1) // batch_size
        batch_pbar = tqdm(range(0, total_images, batch_size), 
                         desc=f'  {channel} batches', 
                         leave=False, 
                         position=1)
        for batch_idx, batch_start in enumerate(batch_pbar):
            batch_end = min(batch_start + batch_size, total_images)
            batch_paths = all_image_paths[batch_start:batch_end]
            batch_pbar.set_postfix({'batch': f'{batch_idx + 1}/{num_batches}'})
            
            # Load current batch
            batch_images = []
            batch_metadata = []
            for sub_dir, img_path in batch_paths:
                try:
                    img = imread(str(img_path))
                    if len(img.shape) != 2:
                        tqdm.write(f"  Warning: {img_path.name} is not a 2D image, skipping")
                        continue
                    batch_images.append(img)
                    batch_metadata.append((sub_dir, img_path))
                except Exception as e:
                    tqdm.write(f"  Warning: Failed to read {img_path.name}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # Convert to numpy array
            batch_images = np.array(batch_images)
            
            # Apply correction
            try:
                corrected_batch = basic.transform(batch_images)
            except Exception as e:
                tqdm.write(f"  Error: Failed to apply correction to batch: {e}")
                continue
            
            # Save corrected images
            for i, (sub_dir, img_path) in enumerate(batch_metadata):
                out_sub_dir = Path(out_dir) / sub_dir.name
                out_sub_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_sub_dir / img_path.name
                
                # Ensure correct data type (preserve original bit depth)
                corrected_img = corrected_batch[i]
                if batch_images.dtype == np.uint16:
                    corrected_img = np.clip(corrected_img, 0, 65535)
                    corrected_img = corrected_img.astype(np.uint16)
                else:
                    corrected_img = corrected_img.astype(batch_images.dtype)
                
                try:
                    imsave(str(out_path), corrected_img, check_contrast=False)
                except Exception as e:
                    tqdm.write(f"  Warning: Failed to save {out_path.name}: {e}")
            
            # Release batch memory
            del batch_images, corrected_batch
        
        tqdm.write(f"  {channel} channel processing completed")
    
    print("All channels processing completed!")


def main():
    pass


if __name__ == '__main__':
    main()

