"""Intensity readout module for extracting intensity values at detected coordinates."""

import numpy as np
import pandas as pd
import cv2
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


logger = logging.getLogger('readout_intensity')


def read_intensity_raw(image, coordinates, search_radius=0):
    """Read intensity values directly from raw image at specified coordinates.
    
    Parameters
    ----------
    image : np.ndarray
        Raw image (uint16)
    coordinates : np.ndarray
        (N, 2) array of (Y, X) coordinates (can be float, will be rounded)
    search_radius : int
        Radius to search for local maximum (default: 0).
        If > 0, reads the maximum value within the radius.
        
    Returns
    -------
    np.ndarray
        (N,) array of intensity values
    """
    # Convert to float32 for processing
    image_f = image.astype(np.float32, copy=False)
    
    # Apply local maximum search if requested (vectorized)
    if search_radius > 0:
        size = 2 * int(search_radius) + 1
        # maximum_filter is equivalent to local max search in window
        # This is much faster than looping over points for large N
        image_to_read = maximum_filter(image_f, size=size, mode='constant', cval=0)
    else:
        image_to_read = image_f

    coords = np.round(coordinates).astype(np.int32)
    height, width = image.shape
    
    # Clip coordinates to be safe
    y = np.clip(coords[:, 0], 0, height - 1)
    x = np.clip(coords[:, 1], 0, width - 1)
    
    return image_to_read[y, x]


def read_intensity_tophat(image, coordinates, tophat_radius=3, search_radius=1):
    """Read intensity values from top-hat filtered image with local peak search.
    
    This method applies top-hat morphological filtering directly to the raw image,
    and then searches for the maximum intensity within a local window around each
    coordinate. This provides robustness against small registration errors (drift).
    
    Parameters
    ----------
    image : np.ndarray
        Raw image (uint16)
    coordinates : np.ndarray
        (N, 2) array of (Y, X) coordinates (can be float, will be rounded)
    tophat_radius : int
        Top-hat morphological operation radius
    search_radius : int
        Radius to search for local maximum (default: 1, i.e., 3x3 window).
        Set to 0 to disable search (read center pixel only).
        
    Returns
    -------
    np.ndarray
        (N,) array of intensity values
    """
    # Convert to float32 for processing
    image_f = image.astype(np.float32, copy=False)
    
    # Apply top-hat filtering (no Gaussian blur)
    ksz = 2 * int(tophat_radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    tophat_image = cv2.morphologyEx(image_f, cv2.MORPH_TOPHAT, kernel)
    
    # Apply local maximum search if requested (vectorized)
    if search_radius > 0:
        size = 2 * int(search_radius) + 1
        # maximum_filter is equivalent to local max search in window
        image_to_read = maximum_filter(tophat_image, size=size, mode='constant', cval=0)
    else:
        image_to_read = tophat_image
    
    coords = np.round(coordinates).astype(np.int32)
    height, width = image.shape
    
    # Clip coordinates to be safe
    y = np.clip(coords[:, 0], 0, height - 1)
    x = np.clip(coords[:, 1], 0, width - 1)
    
    return image_to_read[y, x]


def read_intensity_integrated(image, coordinates, window_radius=3, use_gaussian_weights=True):
    """Read intensity values using local window integration with weighted averaging.
    
    This method extracts intensity by integrating over a local window around each
    coordinate, with optional Gaussian weighting. This provides better noise
    robustness and can utilize sub-pixel coordinates.
    
    Parameters
    ----------
    image : np.ndarray
        Raw image (uint16)
    coordinates : np.ndarray
        (N, 2) array of (Y, X) coordinates (can be float for sub-pixel precision)
    window_radius : int
        Radius of the integration window (default: 3, matching tophat_radius)
    use_gaussian_weights : bool
        If True, use Gaussian weights (center-weighted). If False, use uniform weights.
        
    Returns
    -------
    np.ndarray
        (N,) array of integrated intensity values
    """
    image = image.astype(np.float32, copy=False)
    coords = np.asarray(coordinates, dtype=np.float32)
    n_points = coords.shape[0]
    
    if n_points == 0:
        return np.array([], dtype=np.float32)
    
    window_radius = int(window_radius)
    window_size = 2 * window_radius + 1
    
    # Pre-compute Gaussian weights if needed
    if use_gaussian_weights:
        # Use sigma = window_radius / 2 for reasonable falloff
        sigma = max(window_radius / 2.0, 0.5)
        y_grid, x_grid = np.meshgrid(
            np.arange(window_size) - window_radius,
            np.arange(window_size) - window_radius,
            indexing='ij'
        )
        weights = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        weights = weights / np.sum(weights)  # Normalize
    else:
        weights = np.ones((window_size, window_size), dtype=np.float32)
        weights = weights / np.sum(weights)
    
    intensities = np.zeros(n_points, dtype=np.float32)
    height, width = image.shape
    
    # Process each coordinate
    for i in range(n_points):
        y_center, x_center = coords[i, 0], coords[i, 1]
        
        # Calculate integer bounds
        y_min = int(np.floor(y_center - window_radius))
        y_max = int(np.ceil(y_center + window_radius)) + 1
        x_min = int(np.floor(x_center - window_radius))
        x_max = int(np.ceil(x_center + window_radius)) + 1
        
        # Check bounds
        if (y_min < 0 or y_max > height or x_min < 0 or x_max > width):
            # Use bilinear interpolation for edge cases
            if (0 <= y_center < height and 0 <= x_center < width):
                # Simple bilinear interpolation at center
                y0, y1 = int(np.floor(y_center)), int(np.ceil(y_center))
                x0, x1 = int(np.floor(x_center)), int(np.ceil(x_center))
                y0, y1 = np.clip([y0, y1], 0, height - 1)
                x0, x1 = np.clip([x0, x1], 0, width - 1)
                
                dy = y_center - y0
                dx = x_center - x0
                
                val = (image[y0, x0] * (1 - dx) * (1 - dy) +
                       image[y0, x1] * dx * (1 - dy) +
                       image[y1, x0] * (1 - dx) * dy +
                       image[y1, x1] * dx * dy)
                intensities[i] = val
            else:
                intensities[i] = 0.0
            continue
        
        # Extract local window
        window = image[y_min:y_max, x_min:x_max].copy()
        
        # Adjust weights if window is smaller than expected (edge case)
        actual_h, actual_w = window.shape
        if actual_h < window_size or actual_w < window_size:
            # Crop weights to match actual window size
            w_h_start = (window_size - actual_h) // 2
            w_w_start = (window_size - actual_w) // 2
            w_cropped = weights[w_h_start:w_h_start+actual_h, w_w_start:w_w_start+actual_w]
            w_cropped = w_cropped / np.sum(w_cropped)  # Renormalize
            intensities[i] = np.sum(window * w_cropped)
        else:
            # Standard case: compute weighted sum
            intensities[i] = np.sum(window * weights)
    
    return intensities


def read_intensity_gaussian_fit(image, coordinates, sigma=1.5, fit_radius=3):
    """Read intensity values using 2D Gaussian fitting.
    
    This method fits a 2D Gaussian function to each spot to extract background
    values. This provides more accurate background measurements, especially for
    overlapping or closely spaced spots.
    
    Model: I(x,y) = A * exp(-((x-x0)²+(y-y0)²)/(2σ²)) + B
    where A is the amplitude (signal intensity), B is the background.
    
    Parameters
    ----------
    image : np.ndarray
        Raw image (uint16)
    coordinates : np.ndarray
        (N, 2) array of (Y, X) coordinates (can be float)
    sigma : float
        Expected spot size (Gaussian sigma). Default 1.5 based on tophat_radius=3.
    fit_radius : int
        Radius of region around each coordinate to use for fitting (default: 3)
        
    Returns
    -------
    np.ndarray
        (N,) array of fitted background values (B in the model)
    """
    from scipy.optimize import curve_fit
    
    image = image.astype(np.float32, copy=False)
    coords = np.asarray(coordinates, dtype=np.float32)
    n_points = coords.shape[0]
    
    if n_points == 0:
        return np.array([], dtype=np.float32)
    
    fit_radius = int(fit_radius)
    window_size = 2 * fit_radius + 1
    height, width = image.shape
    
    # Define 2D Gaussian function
    def gaussian_2d(params, y, x):
        """2D Gaussian: A * exp(-((x-x0)²+(y-y0)²)/(2σ²)) + B"""
        A, x0, y0, sigma_fit, B = params
        return A * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma_fit**2)) + B
    
    def gaussian_2d_flat(params, y_flat, x_flat):
        """Flattened version for curve_fit"""
        return gaussian_2d(params, y_flat, x_flat)
    
    intensities = np.zeros(n_points, dtype=np.float32)
    
    # Process each coordinate
    for i in range(n_points):
        y_center, x_center = coords[i, 0], coords[i, 1]
        
        # Calculate integer bounds for fitting region
        y_min = max(0, int(np.floor(y_center - fit_radius)))
        y_max = min(height, int(np.ceil(y_center + fit_radius)) + 1)
        x_min = max(0, int(np.floor(x_center - fit_radius)))
        x_max = min(width, int(np.ceil(x_center + fit_radius)) + 1)
        
        # Check if region is too small
        if (y_max - y_min < 3) or (x_max - x_min < 3):
            # Fallback: simple bilinear interpolation at center
            y0, y1 = int(np.floor(y_center)), int(np.ceil(y_center))
            x0, x1 = int(np.floor(x_center)), int(np.ceil(x_center))
            y0, y1 = np.clip([y0, y1], 0, height - 1)
            x0, x1 = np.clip([x0, x1], 0, width - 1)
            dy = y_center - y0
            dx = x_center - x0
            intensities[i] = (image[y0, x0] * (1 - dx) * (1 - dy) +
                            image[y0, x1] * dx * (1 - dy) +
                            image[y1, x0] * (1 - dx) * dy +
                            image[y1, x1] * dx * dy)
            continue
        
        # Extract local region
        region = image[y_min:y_max, x_min:x_max].copy()
        region_h, region_w = region.shape
        
        # Create coordinate grids (relative to image origin)
        y_grid, x_grid = np.meshgrid(
            np.arange(y_min, y_max),
            np.arange(x_min, x_max),
            indexing='ij'
        )
        
        # Flatten for curve_fit
        y_flat = y_grid.flatten()
        x_flat = x_grid.flatten()
        data_flat = region.flatten()
        
        # Initial parameter estimates
        peak_val = float(np.max(region))
        # Estimate background from edge pixels (edges are more likely to be background)
        # Use border pixels (1-pixel border) to estimate background
        if region_h > 2 and region_w > 2:
            border_pixels = np.concatenate([
                region[0, :],      # Top edge
                region[-1, :],     # Bottom edge
                region[:, 0],     # Left edge
                region[:, -1]     # Right edge
            ])
            background_est = float(np.mean(border_pixels))
        else:
            # Fallback to minimum if region is too small
            background_est = float(np.min(region))
        
        initial_params = [
            peak_val - background_est,  # A (amplitude)
            x_center,                    # x0
            y_center,                    # y0
            sigma,                       # sigma
            background_est               # B (background) - initial guess, will be refined by fitting
        ]
        bounds = (
            [0, x_min, y_min, 0.1, 0],           # Lower bounds
            [peak_val * 2, x_max, y_max, sigma * 3, peak_val]  # Upper bounds
        )
        
        # Perform fitting
        popt, _ = curve_fit(
            gaussian_2d_flat,
            (y_flat, x_flat),
            data_flat,
            p0=initial_params,
            bounds=bounds,
            maxfev=100,  # Limit iterations for speed
            method='trf'  # Trust Region Reflective algorithm
        )
        
        # Extract background (B is the 5th parameter, index 4) and Amplitude (A is index 0)
        # Return Amplitude (A) which represents the signal intensity
        intensities[i] = max(0.0, float(popt[0]))
    
    return intensities


# Dictionary mapping method names to functions
INTENSITY_READ_METHODS = {
    'raw': read_intensity_raw,
    'tophat': read_intensity_tophat,
    'integrated': read_intensity_integrated,
    'gaussian_fit': read_intensity_gaussian_fit,
}


def _read_intensity_for_single_image(args):
    """Helper function for parallel intensity reading (must be at module level for pickle).
    
    Parameters
    ----------
    args : tuple
        (tile_path, channel, cyc, coordinates, intensity_method, method_kwargs)
        tile_path is a string path to the image file (or None if in cache)
        coordinates is a numpy array of (Y, X) coordinates
        intensity_method is the method name ('raw', 'tophat', 'gaussian_fit', etc.)
        method_kwargs is a dict of additional parameters for the read method
        
    Returns
    -------
    tuple
        (cyc, channel, intensities, error_message)
        intensities is a numpy array or None if error
    """
    tile_path, channel, cyc, coordinates, intensity_method, method_kwargs = args
    # Load raw image if path provided
    if tile_path is not None:
        from skimage.io import imread
        raw_image = imread(str(tile_path))
    else:
        # This shouldn't happen, but handle gracefully
        return (cyc, channel, None, "No image path provided")
    
    # Get the read function
    read_func = INTENSITY_READ_METHODS.get(intensity_method)
    if read_func is None:
        return (cyc, channel, None, f"Unknown intensity method: {intensity_method}")
    
    # Read intensities using specified method
    intensities = read_func(raw_image, coordinates, **method_kwargs)
    return (cyc, channel, intensities, None)


def get_intensity_df_for_tile(registered_dir, tile_name, coordinates, channels, cyc_num, 
                               feature_cache=None, max_workers=8, intensity_method='raw', **method_kwargs):
    """Build an intensity dataframe for provided coordinates in a single tile using parallel processing.
    
    Parameters
    ----------
    registered_dir : str or Path
        Base directory containing cyc_n_chn folders
    tile_name : str
        Tile file name (e.g., 'FocalStack_172.tif')
    coordinates : np.ndarray
        Nx2 array-like of (Y,X) coordinates in tile-local coordinates (can be float)
    channels : list
        List of channel names
    cyc_num : int
        Number of cycles to read
    feature_cache : dict, optional
        Cache of feature images {(cyc, channel): feature_image}
        Note: For intensity reading, we need raw images, not feature images.
        This cache is kept for backward compatibility but not used for intensity reading.
    max_workers : int
        Maximum number of parallel workers for intensity reading (default: 8)
    intensity_method : str
        Intensity reading method: 'raw', 'tophat', 'integrated', 'gaussian_fit' (default: 'raw')
    **method_kwargs
        Additional parameters for intensity reading method
        For 'tophat': tophat_radius
        For 'integrated': window_radius, use_gaussian_weights
        For 'gaussian_fit': sigma, fit_radius
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Y','X', 'cyc_1_cy3', ...]
    """
    registered_dir = Path(registered_dir)
    coords = np.asarray(coordinates)
    
    # If no coordinates, return an empty dataframe with the expected columns
    cols = ['Y', 'X'] + [f'cyc_{c}_{ch}' for c in range(1, cyc_num + 1) for ch in channels]
    if coords.size == 0:
        return pd.DataFrame(columns=cols)

    intensity_df = pd.DataFrame({'Y': coords[:, 0], 'X': coords[:, 1]})
    
    # Get the read function
    read_func = INTENSITY_READ_METHODS.get(intensity_method)
    if read_func is None:
        raise ValueError(f"Unknown intensity method: {intensity_method}")
    
    # Collect tasks for parallel processing
    # Note: We always need raw images for intensity reading, regardless of feature_cache
    intensity_tasks = []
    
    for cyc in range(1, cyc_num + 1):
        for channel in channels:
            cyc_chn_dir = registered_dir / f'cyc_{cyc}_{channel}'
            tile_path = cyc_chn_dir / tile_name
            if tile_path.exists():
                intensity_tasks.append((str(tile_path), channel, cyc, coords, intensity_method, method_kwargs))
            else:
                # Image doesn't exist, will be set to NaN
                intensity_df[f'cyc_{cyc}_{channel}'] = np.nan
    
    # Parallel intensity reading
    if intensity_tasks:
        n_workers = min(len(intensity_tasks), max_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_read_intensity_for_single_image, task) for task in intensity_tasks]
            for future in futures:
                cyc, channel, intensities, error = future.result()
                if error is None and intensities is not None:
                    intensity_df[f'cyc_{cyc}_{channel}'] = intensities
                else:
                    logger.warning(f"Failed to read intensity for cyc_{cyc}_{channel}: {error}")
                    intensity_df[f'cyc_{cyc}_{channel}'] = np.nan

    return intensity_df


def extract_intensity_for_tile(registered_dir, tile_name, out_directory, channels, cycle_num, seq_cycle,
                                snrs, min_intensity_threshold, verbose=False,
                                detection_method='spotiflow', intensity_method='tophat',
                                max_workers=8,
                                **method_kwargs):
    """
    Extract intensity data for detected coordinates in a single tile.
    
    This function performs spot detection and intensity readout only.
    Signal correction and base calling are handled separately in coordinate_transform.py.

    Parameters
    ----------
    registered_dir : str or Path
        Base directory containing cyc_n_chn folders
    tile_name : str
        Tile file name (e.g., 'FocalStack_172.tif')
    out_directory : str or Path
        Directory to write output CSV
    channels : list
        List of channel names
    cycle_num : int
        Number of cycles to use for spot detection
    seq_cycle : int
        Number of cycles to read for intensity extraction
    snrs : dict
        Dictionary mapping channel names to SNR thresholds (used for traditional methods only)
    min_intensity_threshold : float
        Minimum intensity threshold for conservative filtering
        Only spots with at least one cycle/channel above this threshold are kept.
        This is a conservative filter to remove obvious noise while preserving
        potentially recoverable weak signals. After global correction in 
        coordinate_transform.py, these signals may become detectable.
        Set to 0 to disable filtering (not recommended for large datasets).
    verbose : bool
        Whether to show progress bars and detailed output (default: False)
    detection_method : str
        Spot detection method: 'gaussian_tophat', 'gaussian_dog', 'tophat', 'dog', or 'spotiflow' (default: 'spotiflow')
    intensity_method : str
        Intensity reading method: 'raw', 'tophat', 'integrated', 'gaussian_fit' (default: 'raw')
    max_workers : int
        Number of worker processes for internal parallelization (detection/reading within tile).
        Set to 1 when running in an external process pool to avoid nested pool issues.
    **method_kwargs
        Additional parameters for detection and intensity methods
        For intensity 'tophat': tophat_radius
        For intensity 'integrated': window_radius, use_gaussian_weights
        For detection 'gaussian_tophat': sigma, tophat_radius
        For detection 'dog': sigma1, sigma2, normalize_percentile
        For detection 'gaussian_dog': sigma, sigma1, sigma2, normalize_percentile
        For intensity 'gaussian_fit': sigma, fit_radius, estimate_background

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Y', 'X', 'cyc_1_cy3', 'cyc_1_cy5', ...]
        Returns empty DataFrame if no coordinates found
    """
    import os
    from .spot_detection import get_coordinates_for_tile
    
    os.makedirs(out_directory, exist_ok=True)

    # Stage 1: Parallel preprocessing + detection (8 images: 4 cycles × 2 channels)
    # This returns both coordinates and preprocessed images for reuse
    if verbose:
        print(f'Extracting coordinates for tile {tile_name}...')
    # Extract detection method kwargs
    detection_kwargs = {}
    intensity_kwargs = {}
    if detection_method == 'spotiflow':
        detection_kwargs = {
            'model_path': method_kwargs.get('detection_model_path', None),
            'pretrained_name': method_kwargs.get('detection_pretrained_name', 'general'),
            'device': method_kwargs.get('detection_device', 'cuda'),
            'prob_thresh': method_kwargs.get('detection_prob_thresh', 0.2),
            'subpix': method_kwargs.get('detection_subpix', False),
            'verbose': method_kwargs.get('detection_verbose', False)
        }
    elif detection_method == 'gaussian_tophat':
        detection_kwargs = {
            'sigma': method_kwargs.get('detection_sigma', 1.0),
            'tophat_radius': method_kwargs.get('detection_tophat_radius', 3)
        }
    elif detection_method == 'gaussian_dog':
        detection_kwargs = {
            'sigma': method_kwargs.get('detection_sigma', 1.0),
            'sigma1': method_kwargs.get('detection_sigma1', 1.2),
            'sigma2': method_kwargs.get('detection_sigma2', 2.5),
            'normalize_percentile': method_kwargs.get('detection_normalize_percentile', 99.5)
        }
    
    # Extract search radius
    # For tophat, default to 1 (as requested). For raw, default to 0 (no search) unless specified.
    default_search_radius = 1 if intensity_method == 'tophat' else 0
    search_radius = method_kwargs.get('intensity_search_radius', default_search_radius)

    if intensity_method == 'tophat':
        intensity_kwargs = {
            'tophat_radius': method_kwargs.get('intensity_tophat_radius', 3),
            'search_radius': search_radius
        }
    elif intensity_method == 'raw':
        intensity_kwargs = {
            'search_radius': search_radius
        }
    elif intensity_method == 'integrated':
        intensity_kwargs = {
            'window_radius': method_kwargs.get('intensity_window_radius', 3),
            'use_gaussian_weights': method_kwargs.get('intensity_use_gaussian_weights', True)
        }
    elif intensity_method == 'gaussian_fit':
        intensity_kwargs = {
            'sigma': method_kwargs.get('intensity_sigma', 1.5),
            'fit_radius': method_kwargs.get('intensity_fit_radius', 3)
        }
    
    coordinates, preprocessed_cache = get_coordinates_for_tile(
        registered_dir, tile_name, channels, cycle_num, snrs, 
        max_workers=max_workers, method=detection_method, **detection_kwargs
    )
    if verbose:
        print(f'Extracted {coordinates.shape[0]} puncta for tile {tile_name}.')

    if coordinates.size == 0:
        # Write empty intensity file
        empty_int_df = pd.DataFrame(columns=['Y', 'X'] + [f'cyc_{c}_{ch}' for c in range(1, seq_cycle + 1) for ch in channels])
        tile_stem = Path(tile_name).stem
        empty_int_df.to_csv(os.path.join(out_directory, f'{tile_stem}_intensity.csv'), index=False)
        if verbose:
            print(f'No coordinates found for tile {tile_name}; wrote empty intensity file.')
        return empty_int_df

    # Stage 2: Parallel intensity reading
    # Note: We read from raw images, not from feature images in cache
    if verbose:
        print(f'Building intensity dataframe for tile {tile_name}...')
    intensity_df = get_intensity_df_for_tile(
        registered_dir, tile_name, coordinates, channels, cyc_num=seq_cycle, 
        feature_cache=preprocessed_cache, max_workers=max_workers, 
        intensity_method=intensity_method, **intensity_kwargs
    )
    
    # Conservative filtering: Remove spots with all intensities below threshold
    # This is a conservative filter to reduce storage/IO while preserving potentially
    # recoverable signals. After global correction, weak signals may become detectable.
    if min_intensity_threshold > 0:
        intensity_max = intensity_df.iloc[:, 2:].max(axis=1)
        n_before = len(intensity_df)
        intensity_df = intensity_df[intensity_max >= min_intensity_threshold]
        n_after = len(intensity_df)
        if verbose:
            print(f'  Filtered {n_before - n_after} spots below threshold ({min_intensity_threshold}), '
                  f'kept {n_after} spots ({n_after/n_before*100:.1f}%)')
    
    # Deduplicate spots within search_radius
    if search_radius > 0 and len(intensity_df) > 0:
        if verbose:
            print(f'  Deduplicating spots within radius {search_radius} (Chebyshev distance)...')
        
        # 1. Calculate scores (max intensity across all channels/cycles)
        # Use columns starting with 'cyc_'
        intensity_cols = [c for c in intensity_df.columns if c.startswith('cyc_')]
        if intensity_cols:
            scores = intensity_df[intensity_cols].max(axis=1).fillna(-1).values
        else:
            scores = np.zeros(len(intensity_df))
            
        coords = intensity_df[['Y', 'X']].values
        
        # 2. Build KDTree
        tree = cKDTree(coords)
        
        # 3. Find pairs within radius (Chebyshev for square window match)
        # p=np.inf corresponds to Chebyshev distance (box region)
        pairs = tree.query_pairs(r=search_radius, p=float('inf'))
        
        if pairs:
            # 4. Find connected components
            n_points = len(intensity_df)
            rows = [p[0] for p in pairs]
            cols = [p[1] for p in pairs]
            # Create symmetric adjacency matrix
            data = np.ones(len(pairs), dtype=bool)
            adj = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
            # connected_components returns (n_components, labels)
            n_components, labels = connected_components(adj, directed=False)
            
            # 5. Select best spot per component
            df_dedup_helper = pd.DataFrame({
                'label': labels,
                'score': scores,
                'orig_idx': intensity_df.index
            })
            
            # Find index of max score per label
            best_indices = df_dedup_helper.loc[df_dedup_helper.groupby('label')['score'].idxmax(), 'orig_idx']
            
            n_before_dedup = len(intensity_df)
            intensity_df = intensity_df.loc[best_indices].sort_index()
            n_after_dedup = len(intensity_df)
            
            if verbose:
                print(f'  Removed {n_before_dedup - n_after_dedup} duplicate spots (kept {n_after_dedup}).')

    tile_stem = Path(tile_name).stem
    
    # Save raw intensity data
    # Signal correction and base calling will be applied globally after coordinate transformation
    intensity_df.to_csv(os.path.join(out_directory, f'{tile_stem}_intensity.csv'), index=False)
    if verbose:
        print(f'Saved raw intensity data for {len(intensity_df)} puncta in tile {tile_name}.')
        print(f'  Note: Signal correction and base calling will be applied globally in coordinate_transform.py.')
    
    return intensity_df

