"""Spot detection module for traditional and deep learning-based detection methods."""

import numpy as np
import cv2
import logging
import gc
from pathlib import Path
from skimage.io import imread
from skimage.feature import blob_log
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger('readout_spot_detection')

# Global cache for Spotiflow model to avoid reloading in every call/process
_SPOTIFLOW_MODEL = None

def _get_spotiflow_model(model_path=None, pretrained_name='general', verbose=False):
    """Get or load the global Spotiflow model instance."""
    global _SPOTIFLOW_MODEL
    if _SPOTIFLOW_MODEL is None:
        try:
            from spotiflow.model import Spotiflow
        except ImportError as e:
            raise ImportError(
                "Spotiflow is required for method='spotiflow'. "
                "Install with: pip install sprintseq[detection-dl]"
            ) from e

        if verbose:
            logger.info("Loading Spotiflow model (first time initialization)...")

        if model_path is not None:
            _SPOTIFLOW_MODEL = Spotiflow.from_folder(str(model_path))
        else:
            _SPOTIFLOW_MODEL = Spotiflow.from_pretrained(pretrained_name, verbose=verbose)
    return _SPOTIFLOW_MODEL



def feature_dog(image, sigma1=1.2, sigma2=2.5, normalize_percentile=99.9):
    """Extract features using Difference of Gaussians (DoG) only (no pre-smoothing).
    
    This method computes DoG directly on the normalized image without pre-smoothing,
    keeps only positive responses, normalizes with robust percentile, and scales
    to uint16 range for threshold calculation.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (uint16)
    sigma1 : float
        First Gaussian sigma (smaller, for fine detail)
    sigma2 : float
        Second Gaussian sigma (larger, for background)
    normalize_percentile : float
        Percentile for robust normalization (0-100)
        
    Returns
    -------
    np.ndarray
        Feature image (float32, scaled to uint16 range [0, 65535])
    """
    # Normalize input to [0, 1] range
    img_norm = image.astype(np.float32) / 65535.0
    
    # DoG: Difference of Gaussians (no pre-smoothing)
    g1 = cv2.GaussianBlur(img_norm, ksize=(0, 0), sigmaX=sigma1, sigmaY=sigma1)
    g2 = cv2.GaussianBlur(img_norm, ksize=(0, 0), sigmaX=sigma2, sigmaY=sigma2)
    dog_map = g1 - g2
    
    # Keep only positive responses
    dog_map[dog_map < 0] = 0.0
    
    # Robust normalization to [0, 1]
    p = np.percentile(dog_map, normalize_percentile)
    if p <= 0:
        return np.zeros_like(dog_map, dtype=np.float32)
    dog_map = np.clip(dog_map / p, 0.0, 1.0).astype(np.float32)
    
    # Scale to uint16 range for threshold calculation (matching ImageJ behavior)
    dog_map_scaled = (dog_map * 65535.0).astype(np.float32)
    
    return dog_map_scaled


def feature_gaussian_dog(image, sigma=1, sigma1=1.2, sigma2=2.5, normalize_percentile=99.9):
    """Extract features using Difference of Gaussians (DoG) method.
    
    This method uses DoG (approximating Laplacian of Gaussian) to detect
    spot-like signals. It applies pre-smoothing, then computes DoG, keeps
    only positive responses, normalizes with robust percentile, and scales
    to uint16 range for threshold calculation.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (uint16)
    sigma1 : float
        First Gaussian sigma (smaller, for fine detail)
    sigma2 : float
        Second Gaussian sigma (larger, for background)
    normalize_percentile : float
        Percentile for robust normalization (0-100)
        
    Returns
    -------
    np.ndarray
        Feature image (float32, scaled to uint16 range [0, 65535])
    """
    # Normalize input to [0, 1] range
    img_norm = image.astype(np.float32) / 65535.0
    
    # Pre-smoothing
    img_s = cv2.GaussianBlur(img_norm, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    
    # DoG: Difference of Gaussians
    g1 = cv2.GaussianBlur(img_s, ksize=(0, 0), sigmaX=sigma1, sigmaY=sigma1)
    g2 = cv2.GaussianBlur(img_s, ksize=(0, 0), sigmaX=sigma2, sigmaY=sigma2)
    dog_map = g1 - g2
    
    # Keep only positive responses
    dog_map[dog_map < 0] = 0.0
    
    # Robust normalization to [0, 1]
    p = np.percentile(dog_map, normalize_percentile)
    if p <= 0:
        return np.zeros_like(dog_map, dtype=np.float32)
    dog_map = np.clip(dog_map / p, 0.0, 1.0).astype(np.float32)
    
    # Scale to uint16 range for threshold calculation (matching ImageJ behavior)
    dog_map_scaled = (dog_map * 65535.0).astype(np.float32)
    
    return dog_map_scaled


def feature_tophat(image, tophat_radius=3):
    """Extract features using top-hat filtering only (no Gaussian blur).
    
    This method applies top-hat morphological filtering directly to the image
    to enhance spot-like features and suppress background, without pre-smoothing.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (uint16)
    tophat_radius : int
        Top-hat morphological operation radius
        
    Returns
    -------
    np.ndarray
        Feature image (uint16)
    """
    # Convert to float32 for processing
    image_f = image.astype(np.float32, copy=False)
    
    # Apply top-hat filtering directly (no Gaussian blur)
    ksz = 2 * int(tophat_radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    processed = cv2.morphologyEx(image_f, cv2.MORPH_TOPHAT, kernel)
    
    # Convert back to uint16
    out_image = np.rint(processed).astype(np.int64)
    np.clip(out_image, 0, 65535, out=out_image)
    out_image = out_image.astype(np.uint16)
    
    return out_image


def feature_gaussian_tophat(image, sigma=1.0, tophat_radius=3):
    """Extract features using Gaussian blur + top-hat filtering.
    
    This method applies Gaussian smoothing followed by top-hat morphological
    filtering to enhance spot-like features and suppress background.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (uint16)
    sigma : float
        Gaussian blur sigma
    tophat_radius : int
        Top-hat morphological operation radius
        
    Returns
    -------
    np.ndarray
        Feature image (uint16)
    """
    # Convert to float32 for processing
    image_f = image.astype(np.float32, copy=False)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image_f, ksize=(0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)

    # Apply top-hat filtering if specified
    if tophat_radius and tophat_radius > 0:
        ksz = 2 * int(tophat_radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        processed = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
    else:
        processed = blurred

    # Convert back to uint16
    out_image = np.rint(processed).astype(np.int64)
    np.clip(out_image, 0, 65535, out=out_image)
    out_image = out_image.astype(np.uint16)
    
    return out_image


# Backward compatibility alias
preprocess_image_in_memory = feature_gaussian_dog


def blob_log_detection(image, min_sigma=1.0, max_sigma=2.0, num_sigma=2,
                       threshold=0.005, overlap=0.5):
    """End-to-end LoG (Laplacian of Gaussian) spot detection via skimage.feature.blob_log.

    skimage.feature.blob_log performs scale-space LoG filtering at num_sigma
    intermediate sigmas in [min_sigma, max_sigma], finds local maxima of the
    scale-normalized response, and merges overlapping blobs by area-overlap NMS.
    The result is a set of (y, x, sigma) blob descriptors; this wrapper returns
    only (y, x) coordinates so it slots into the same return contract as the
    other methods in get_spot_coordinates.

    Approximates the Fiji TrackMate LogDetector (LogDetector.java). For
    TrackMate's `radius` parameter on a 2D image, sigma ~= radius / sqrt(2)
    (e.g. radius=2.0 -> sigma~=1.41). TrackMate's threshold is in raw image
    units; skimage.blob_log applies threshold on the scale-normalized LoG
    response of the float-converted image, so plan to retune empirically.

    Parameters
    ----------
    image : np.ndarray
        Input image. blob_log calls img_as_float internally
        (uint16 -> /65535, float assumed in [0, 1]).
    min_sigma, max_sigma : float
        Bounds of the Gaussian sigma swept by blob_log.
    num_sigma : int
        Number of intermediate sigmas, >= 1. Use 1 for a single-scale LoG.
    threshold : float
        Absolute lower bound on the scale-normalized LoG response.
    overlap : float
        NMS overlap threshold; blobs whose pairwise area-overlap exceeds
        this are merged (smaller-sigma blob discarded).

    Returns
    -------
    np.ndarray
        (N, 2) array of (Y, X) coordinates as float32. Empty (0, 2) if none.
    """
    blobs = blob_log(
        image,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap,
    )
    if blobs is None or len(blobs) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return blobs[:, :2].astype(np.float32, copy=False)


def find_maxima(image, tolerance, threshold=None, strict=False, exclude_on_edges=False, min_distance=2):
    """Find maxima using ImageJ MaximumFinder algorithm (exact implementation).
    
    This is a direct port of ImageJ's MaximumFinder.findMaxima() method.
    It uses flood-fill based prominence checking, not H-Maxima transform.
    
    Algorithm:
    1. Find all local maxima (8-neighborhood check)
    2. Sort maxima by intensity (highest first)
    3. For each maximum (starting from highest):
       - Flood fill to all pixels >= (peak_value - tolerance)
       - If flood fill reaches a PROCESSED pixel, this peak is eliminated
       - If flood fill reaches a higher peak (within tolerance), handle sorting error
       - If peak survives, mark it as valid
    
    Parameters
    ----------
    image : np.ndarray
        Input image (2D array, float32 or uint16)
    tolerance : float
        Prominence requirement (ImageJ's "Prominence" parameter).
        A peak is only accepted if it protrudes more than this value
        from the ridge to a higher maximum. This is an ABSOLUTE height difference.
    threshold : float, optional
        Minimum height threshold. Pixels below this value are ignored.
        If None, no threshold is applied.
    strict : bool
        When False, the global maximum is accepted even if all other pixels
        are less than 'tolerance' below this level.
    exclude_on_edges : bool
        Whether to exclude maxima at image edges
    min_distance : int
        Minimum distance between peaks (applied as post-processing)
        
    Returns
    -------
    np.ndarray
        (N, 2) array of (Y, X) coordinates of detected peaks (float32)
    """
    image = np.asarray(image, dtype=np.float32)
    height, width = image.shape
    
    # Constants matching ImageJ
    DIR_X_OFFSET = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)
    DIR_Y_OFFSET = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int32)
    
    # Pixel type flags (matching ImageJ constants)
    MAXIMUM = 1
    LISTED = 2
    PROCESSED = 4
    EQUAL = 16
    MAX_POINT = 32
    
    # Calculate global min/max
    if threshold is not None:
        valid_mask = image >= threshold
        if not np.any(valid_mask):
            return np.empty((0, 2), dtype=np.float32)
        global_min = float(np.min(image[valid_mask]))
        global_max = float(np.max(image[valid_mask]))
    else:
        global_min = float(np.min(image))
        global_max = float(np.max(image))
    
    maximum_possible = global_max > global_min
    if strict and (global_max - global_min) <= tolerance:
        maximum_possible = False
    
    if not maximum_possible:
        return np.empty((0, 2), dtype=np.float32)
    
    # Step 1: Find all local maxima (8-neighborhood check)
    # This corresponds to getSortedMaxPoints in ImageJ
    types = np.zeros((height, width), dtype=np.uint8)  # Pixel type flags
    max_points = []  # List of (value, y, x) tuples
    
    for y in range(height):
        for x in range(width):
            v = image[y, x]
            
            # Skip if at global minimum
            if v == global_min:
                continue
            
            # Skip edges if exclude_on_edges
            if exclude_on_edges and (x == 0 or x == width-1 or y == 0 or y == height-1):
                continue
            
            # Apply threshold
            if threshold is not None and v < threshold:
                continue
            
            # Check if this is a local maximum (8-neighborhood)
            is_max = True
            for d in range(8):
                x2 = x + DIR_X_OFFSET[d]
                y2 = y + DIR_Y_OFFSET[d]
                
                # Check bounds
                if x2 < 0 or x2 >= width or y2 < 0 or y2 >= height:
                    continue
                
                v_neighbor = image[y2, x2]
                if v_neighbor > v:
                    is_max = False
                    break
            
            if is_max:
                types[y, x] = MAXIMUM
                max_points.append((v, y, x))
    
    if len(max_points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    
    # Sort maxima by value (highest first)
    max_points.sort(reverse=True, key=lambda x: x[0])
    
    # Step 2: Analyze maxima using flood-fill (analyzeAndMarkMaxima in ImageJ)
    # Process from highest to lowest
    valid_maxima = []
    p_list = []  # Flood fill queue
    
    for peak_idx, (v0, y0, x0) in enumerate(max_points):
        # Skip if already processed (reached from a higher peak)
        if (types[y0, x0] & PROCESSED) != 0:
            continue
        
        # Flood fill from this peak
        sorting_error = False
        max_possible = True
        
        # Track equal-height points for finding center
        x_equal = float(x0)
        y_equal = float(y0)
        n_equal = 1
        
        do_retry = True
        while do_retry:
            do_retry = False
            p_list = [(y0, x0)]
            types[y0, x0] |= (EQUAL | LISTED)
            list_i = 0
            is_edge_maximum = (x0 == 0 or x0 == width-1 or y0 == 0 or y0 == height-1)
            sorting_error = False
            max_possible = True
            
            # Flood fill: expand to all neighbors within tolerance
            while list_i < len(p_list):
                y, x = p_list[list_i]
                
                # Check all 8 neighbors
                for d in range(8):
                    x2 = x + DIR_X_OFFSET[d]
                    y2 = y + DIR_Y_OFFSET[d]
                    
                    # Check bounds
                    if x2 < 0 or x2 >= width or y2 < 0 or y2 >= height:
                        continue
                    
                    # Skip if already in list
                    if (types[y2, x2] & LISTED) != 0:
                        continue
                    
                    v2 = image[y2, x2]
                    
                    # If reached a processed pixel, this peak is eliminated
                    if (types[y2, x2] & PROCESSED) != 0:
                        max_possible = False
                        break
                    
                    # If reached a higher point, this is not a maximum
                    if v2 > v0:
                        max_possible = False
                        break
                    
                    # If within tolerance, add to flood fill
                    if v2 >= v0 - tolerance:
                        # Check for sorting error (higher point within tolerance)
                        if v2 > v0:
                            sorting_error = True
                            v0 = v2
                            x0 = x2
                            y0 = y2
                            do_retry = True
                            break
                        
                        p_list.append((y2, x2))
                        types[y2, x2] |= LISTED
                        
                        # Check edge
                        if (x2 == 0 or x2 == width-1 or y2 == 0 or y2 == height-1):
                            is_edge_maximum = True
                            if exclude_on_edges and (strict or v2 >= v0):
                                max_possible = False
                                break
                        
                        # Track equal-height points
                        if v2 == v0:
                            types[y2, x2] |= EQUAL
                            x_equal += x2
                            y_equal += y2
                            n_equal += 1
                
                if not max_possible or do_retry:
                    break
                
                list_i += 1
            
            # Handle sorting error: reset and retry
            if sorting_error:
                for y, x in p_list:
                    types[y, x] = 0
                continue
            
            # Mark all points in flood fill as processed
            for y, x in p_list:
                types[y, x] &= ~(LISTED | EQUAL)
                types[y, x] |= PROCESSED
            
            # If this is a valid maximum, find the center point
            if max_possible:
                x_equal /= n_equal
                y_equal /= n_equal
                
                # Find point closest to center among equal-height points
                min_dist2 = float('inf')
                best_y, best_x = y0, x0
                for y, x in p_list:
                    if (types[y, x] & EQUAL) != 0:
                        dist2 = (x - x_equal)**2 + (y - y_equal)**2
                        if dist2 < min_dist2:
                            min_dist2 = dist2
                            best_y, best_x = y, x
                
                # Mark as MAX_POINT
                types[best_y, best_x] |= MAX_POINT
                
                # Add to valid maxima (if not excluded by edge rule)
                if not (exclude_on_edges and is_edge_maximum):
                    valid_maxima.append((best_y, best_x))
        
        # Reset LISTED flags for next iteration
        for y, x in p_list:
            types[y, x] &= ~LISTED
    
    if len(valid_maxima) == 0:
        return np.empty((0, 2), dtype=np.float32)
    
    coords = np.array(valid_maxima, dtype=np.float32)
    
    # Step 3: Apply min_distance filtering (post-processing)
    if min_distance > 1 and len(coords) > 1:
        # Sort by intensity (descending)
        intensities = image[coords[:, 0].astype(int), coords[:, 1].astype(int)]
        sorted_indices = np.argsort(intensities)[::-1]
        coords_sorted = coords[sorted_indices]
        
        # Use KD-Tree for efficient filtering
        try:
            from scipy.spatial import cKDTree
            
            kept_indices = [0]
            kept_coords_list = [coords_sorted[0].tolist()]
            tree = None
            tree_size = 0
            rebuild_interval = max(100, len(coords_sorted) // 100)
            
            for i in range(1, len(coords_sorted)):
                current = coords_sorted[i]
                
                if tree is None or len(kept_coords_list) >= tree_size + rebuild_interval:
                    kept_coords_array = np.array(kept_coords_list)
                    tree = cKDTree(kept_coords_array)
                    tree_size = len(kept_coords_list)
                
                neighbors = tree.query_ball_point(current, r=min_distance - 1e-6)
                if len(neighbors) == 0:
                    kept_indices.append(i)
                    kept_coords_list.append(current.tolist())
            
            coords = coords_sorted[kept_indices]
        except ImportError:
            # Fallback: simple distance check
            kept_indices = [0]
            for i in range(1, len(coords_sorted)):
                current = coords_sorted[i]
                distances = np.sqrt(np.sum((coords_sorted[kept_indices] - current) ** 2, axis=1))
                if np.all(distances >= min_distance):
                    kept_indices.append(i)
            coords = coords_sorted[kept_indices]
    
    return coords


def _extract_features(image, method, **kwargs):
    """Internal helper function to extract features from an image.
    
    This function is used for traditional detection methods that follow a two-stage
    pipeline: feature extraction → peak detection. Deep learning methods like
    'spotiflow' are handled directly in get_spot_coordinates() and bypass this function.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (uint16)
    method : str
        Feature extraction method: 'gaussian_tophat', 'gaussian_dog', 'tophat', or 'dog'
        Note: 'spotiflow' is handled directly in get_spot_coordinates(), not here.
    **kwargs
        Additional parameters passed to the feature extraction function
        
    Returns
    -------
    np.ndarray
        Feature image (float32)
    """
    if method == 'gaussian_tophat':
        feature_image = feature_gaussian_tophat(image, **kwargs)
        feature_image = feature_image.astype(np.float32)
    elif method == 'gaussian_dog':
        feature_image = feature_gaussian_dog(image, **kwargs)
    elif method == 'tophat':
        feature_image = feature_tophat(image, **kwargs)
        feature_image = feature_image.astype(np.float32)
    elif method == 'dog':
        feature_image = feature_dog(image, **kwargs)
    else:
        raise ValueError(f"Unknown feature extraction method: {method}. Note: 'spotiflow' is handled in get_spot_coordinates(), not here.")
    return feature_image


def get_spot_coordinates(image, method='spotiflow', min_distance=2, **kwargs):
    """Detect spot coordinates from an image using specified detection method.

    This is the unified interface for spot detection. It supports both traditional
    two-stage methods (feature extraction → peak detection) and end-to-end methods
    that produce coordinates directly.

    Traditional methods ('gaussian_tophat', 'gaussian_dog', 'tophat', 'dog'):
        - Extract features from the image
        - Apply find_maxima to detect peak coordinates

    End-to-end methods ('spotiflow', 'blob_log'):
        - Perform detection directly from image to coordinates
        - 'spotiflow': deep-learning detector (lazy-imported)
        - 'blob_log': scale-space LoG via skimage.feature.blob_log
          (approximates Fiji TrackMate LogDetector)

    Parameters
    ----------
    image : np.ndarray
        Input image (uint16)
    method : str
        Spot detection method: 'gaussian_tophat', 'gaussian_dog', 'tophat',
        'dog', 'spotiflow', or 'blob_log'.
    min_distance : int
        Minimum distance between peaks (only used by traditional methods'
        find_maxima post-processing; blob_log uses its own overlap NMS).
    **kwargs
        Additional parameters for the detection method.
        For traditional methods: 'snr' sets threshold as snr * image_median.
        For 'spotiflow': 'model_path', 'pretrained_name', 'device', 'prob_thresh'.
        For 'blob_log': 'min_sigma', 'max_sigma', 'num_sigma', 'threshold', 'overlap'.

    Returns
    -------
    np.ndarray
        (N, 2) array of (Y, X) coordinates. Coordinates can be float or int.
    """
    # End-to-end LoG detection via skimage.feature.blob_log
    if method == 'blob_log':
        min_sigma = kwargs.pop('min_sigma', 1.0)
        max_sigma = kwargs.pop('max_sigma', 2.0)
        num_sigma = kwargs.pop('num_sigma', 2)
        threshold = kwargs.pop('threshold', 0.005)
        overlap = kwargs.pop('overlap', 0.5)
        return blob_log_detection(
            image,
            min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma,
            threshold=threshold, overlap=overlap,
        )

    # Handle spotiflow method separately (end-to-end detection)
    if method == 'spotiflow':
        import torch
        
        # Extract spotiflow-specific parameters
        model_path = kwargs.pop('model_path', None)
        pretrained_name = kwargs.pop('pretrained_name', 'general')
        device = kwargs.pop('device', None)
        prob_thresh = kwargs.pop('prob_thresh', None)
        subpix = kwargs.pop('subpix', None)
        verbose = kwargs.pop('verbose', False)
        
        # Get global model instance (lazy loading, shared within process)
        model = _get_spotiflow_model(model_path, pretrained_name, verbose)
        
        # Determine device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Predict spots (Spotiflow handles format conversion and normalization internally)
        with torch.no_grad():
            spots, details = model.predict(
                image,
                prob_thresh=prob_thresh,
                subpix=subpix,
                verbose=verbose,
                min_distance=min_distance,
                device=device,
            )
        
        # Cleanup to prevent OOM
        if device == 'cuda' or (isinstance(device, str) and device.startswith('cuda')):
            torch.cuda.empty_cache()
        gc.collect()

        return spots
    
    # Traditional methods: extract features then find maxima
    # Extract snr from kwargs (used for threshold calculation, not passed to feature extraction)
    snr = kwargs.pop('snr', None)
    
    # Extract features using traditional methods
    feature_image = _extract_features(image, method, **kwargs)
    
    # Calculate image median for threshold
    img_median = float(np.median(feature_image[feature_image > 0])) if np.any(feature_image > 0) else 0.0
    
    # Determine threshold - use snr * image_median if snr is provided
    # Note: snr was already extracted from kwargs above using pop()
    if snr is not None and img_median > 0:
        threshold_abs = snr * img_median
    elif img_median > 0:
        threshold_abs = img_median * 0.1  # Conservative default based on median
    else:
        threshold_abs = float(np.mean(feature_image)) * 0.1  # Fallback to mean
    
    # Detect peaks
    peaks = find_maxima(
        feature_image,
        tolerance=threshold_abs * 0.1,
        threshold=threshold_abs,
        strict=False,
        exclude_on_edges=False,
        min_distance=min_distance
    )
    # Return as float (allows sub-pixel coordinates in future)
    return peaks


def _preprocess_and_detect_for_single_image(args):
    """Helper function for parallel preprocessing + detection (must be at module level for pickle).
    
    This function combines preprocessing and detection in one step to avoid passing large numpy arrays
    between processes. It returns detected coordinates only.
    
    Parameters
    ----------
    args : tuple
        (tile_path, channel, cyc, snrs, method, method_kwargs)
        tile_path is a string path to the image file
        method is the spot detection method name
        method_kwargs is a dict of additional parameters for the detection method
        
    Returns
    -------
    tuple
        (cyc, channel, coords, error_message)
        coords is empty array if error occurred
    """
    tile_path, channel, cyc, snrs, method, method_kwargs = args
    # Load image
    image = imread(str(tile_path))
    
    # Use get_spot_coordinates for detection (reuse common logic)
    # For spotiflow, snr is ignored but we include it for interface compatibility
    if method == 'spotiflow':
        # Spotiflow doesn't use SNR, but we keep the interface consistent
        method_kwargs_with_snr = {**method_kwargs}  # Don't add snr for spotiflow
    else:
        method_kwargs_with_snr = {**method_kwargs, 'snr': snrs[channel]}
    
    peaks = get_spot_coordinates(
        image,
        method=method,
        min_distance=2,
        **method_kwargs_with_snr
    )
    return (cyc, channel, peaks, None)


def get_coordinates_for_tile(registered_dir, tile_name, channels, cycle_num, snrs, 
                              max_workers=8, method='spotiflow', **method_kwargs):
    """Extract coordinates for a single tile from registered images using parallel detection.
    
    This function performs spot detection in parallel across multiple images
    (typically 4 cycles × 2 channels = 8 images), and returns the detected coordinates.
    
    Parameters
    ----------
    registered_dir : str or Path
        Base directory containing cyc_n_chn folders
    tile_name : str
        Tile file name (e.g., 'FocalStack_172.tif')
    channels : list
        List of channel names
    cycle_num : int
        Number of cycles to use for spot detection
    snrs : dict
        Dictionary mapping channel names to SNR thresholds (used for traditional methods only)
    max_workers : int
        Maximum number of parallel workers for detection (default: 8)
    method : str
        Spot detection method: 'gaussian_tophat', 'gaussian_dog', 'tophat', 'dog', or 'spotiflow' (default: 'spotiflow')
    **method_kwargs
        Additional parameters for the detection method:
        - For 'dog': sigma1, sigma2, normalize_percentile
        - For 'gaussian_dog': sigma1, sigma2, normalize_percentile
        - For 'tophat': tophat_radius
        - For 'gaussian_tophat': sigma, tophat_radius
        - For 'spotiflow': model_path, pretrained_name, device, prob_thresh
        
    Returns
    -------
    tuple
        (coordinates, feature_cache)
        coordinates: (N, 2) array of (Y, X) coordinates in tile-local coordinates (float)
        feature_cache: dict (empty, kept for backward compatibility)
    """
    registered_dir = Path(registered_dir)
    
    # Set default method kwargs if not provided
    if method == 'dog':
        default_kwargs = {'sigma': 1.0, 'sigma1': 1.2, 'sigma2': 2.5, 'normalize_percentile': 99.5}
    elif method == 'gaussian_dog':
        default_kwargs = {'sigma': 1.0, 'sigma1': 1.2, 'sigma2': 2.5, 'normalize_percentile': 99.5}
    elif method == 'tophat':
        default_kwargs = {'tophat_radius': 3}
    elif method == 'gaussian_tophat':
        default_kwargs = {'sigma': 1.0, 'tophat_radius': 3}
    else:
        raise ValueError(f"Unknown method: {method}")
    
    final_method_kwargs = {**default_kwargs, **method_kwargs}
    
    # Collect all image paths to process (preprocessing + detection)
    detection_tasks = []
    for cyc in range(1, 1 + cycle_num):
        for channel in channels:
            cyc_chn_dir = registered_dir / f'cyc_{cyc}_{channel}'
            tile_path = cyc_chn_dir / tile_name
            if tile_path.exists():
                detection_tasks.append((str(tile_path), channel, cyc, snrs, method, final_method_kwargs))
    
    # Parallel preprocessing + detection (limited to max_workers)
    collected = []
    feature_cache = {}  # Keep for backward compatibility, but will be empty
    
    if detection_tasks:
        n_workers = min(len(detection_tasks), max_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_preprocess_and_detect_for_single_image, task) for task in detection_tasks]
            for future in futures:
                cyc, channel, coords, error = future.result()
                if error is None:
                    # Collect coordinates
                    if coords.size > 0:
                        collected.append(coords)
                else:
                    logger.warning(f"Failed to preprocess+detect for cyc_{cyc}_{channel}: {error}")

    if collected:
        all_coords = np.vstack(collected)
        # Use np.unique with tolerance for float coordinates
        # For now, round to nearest integer for uniqueness check
        coords_rounded = np.round(all_coords).astype(np.int32)
        _, unique_indices = np.unique(coords_rounded, axis=0, return_index=True)
        coordinates = all_coords[unique_indices]
    else:
        coordinates = np.empty((0, 2), dtype=np.float32)
    
    return coordinates, feature_cache

