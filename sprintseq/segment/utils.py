"""
Utility functions for cell segmentation and RNA assignment.

Segmentation backends:
- CellSAM (DAPI + optional morphology channels)
- Cellpose (DAPI + optional morphology channels)
- DAPI-only nucleus segmentation + KD-tree spot→nucleus assignment (no membrane channel required)

"morphology" here means any membrane/cytoplasm/cell-body marker — FAM in the canonical SPRINTseq
protocol, but can be anything (e.g. wheat-germ agglutinin, CellMask, a poly-A oligo hybridization).
Pass one or more morphology images via `morphology_paths` (list). When more than one is passed,
they are merged into a single "cell body" channel via pixel-wise max before stacking with DAPI.
"""

import pandas as pd
import numpy as np
import tifffile
from pathlib import Path

from sprintseq.readout import mosaic


def load_and_merge_spots(position_file, mapping_file, probability_threshold=0.8):
    """
    Load position and mapping files, merge them, and filter by probability.
    """
    print(f"Loading positions from {position_file}...")
    pos_df = pd.read_csv(position_file)

    print(f"Loading mappings from {mapping_file}...")
    map_df = pd.read_csv(mapping_file)

    print("Merging dataframes...")
    merged_df = pd.merge(pos_df, map_df, on='index', how='inner')

    print(f"Filtering spots with Probability > {probability_threshold}...")
    filtered_df = merged_df[merged_df['Probability'] > probability_threshold].copy()

    print(f"Retained {len(filtered_df)} spots out of {len(merged_df)}.")

    return filtered_df


class _MemmapStack:
    """Virtual (H, W, C_out) multi-channel stack with per-channel backing by 2D memmap arrays.

    Why this class exists: stitched SPRINTseq images are commonly 30k x 30k (or more)
    uint16, i.e. ~1.8 GB per channel on disk. Eagerly materialising a (H, W, 3) combined
    array for CellSAM input would use ~5.4 GB of RAM up-front, before any segmentation
    runs. Instead this class keeps each channel as a `tifffile.memmap` and constructs
    the combined (tile_h, tile_w, C_out) array *only for the requested slice* — exactly
    the access pattern `_run_tiled_inference` wants.

    Supports the minimum ndarray-like interface that `_run_tiled_inference` needs:
      - `.shape` -> (H, W, C_out)
      - `.dtype`
      - `.ndim` = 3
      - 2D slicing `stack[y_slice, x_slice]` -> real (tile_h, tile_w, C_out) ndarray,
        built on demand. Only the slice region is paged in from disk.
    """

    def __init__(self, reference_shape, dtype, channel_assignments, n_out_channels):
        """
        Parameters
        ----------
        reference_shape : tuple (H, W)
        dtype : numpy dtype
        channel_assignments : list of (out_idx, sources)
            Each entry places pixel-wise max of `sources` (a list of 2D ndarrays /
            memmaps) into output channel `out_idx`. A single-source list is fine.
        n_out_channels : int
            Width of the output channel axis (3 for CellSAM, 2 for Cellpose).
        """
        self._ref_shape = reference_shape
        self._dtype = dtype
        # Normalize: every assignment holds a list (possibly length-1), so __getitem__
        # doesn't need to branch on single- vs multi-source at hot-path time.
        self._assignments = [
            (out_idx, list(sources) if isinstance(sources, (list, tuple)) else [sources])
            for out_idx, sources in channel_assignments
        ]
        self._n_out = n_out_channels

    @property
    def shape(self):
        return (*self._ref_shape, self._n_out)

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return 3

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 2):
            raise IndexError(
                f"_MemmapStack supports only 2D slicing stack[y, x]; got {key!r}"
            )
        y_slice, x_slice = key
        h_ref, w_ref = self._ref_shape
        y0, y1, _ = y_slice.indices(h_ref)
        x0, x1, _ = x_slice.indices(w_ref)
        tile = np.zeros((y1 - y0, x1 - x0, self._n_out), dtype=self._dtype)

        for out_idx, sources in self._assignments:
            # sources is always a list (normalized in __init__). Length-1 short-circuits.
            if len(sources) == 1:
                tile[..., out_idx] = sources[0][y_slice, x_slice]
            else:
                acc = np.array(sources[0][y_slice, x_slice], copy=True)
                for extra in sources[1:]:
                    np.maximum(acc, extra[y_slice, x_slice], out=acc)
                tile[..., out_idx] = acc
        return tile


def _as_lazy_source(src):
    """Accept a path to a stitched TIFF, or an already-opened array-like handle.

    The contract is widened rather than replaced: several per-run scripts under
    `SPRINTseq/experiments/` call `prepare_cellsam_input` / `prepare_cellpose_input` with
    explicit paths and must keep working. `auto_detect_*` returns a path on the legacy
    backend and an opened handle for a run stored as one OME-Zarr store, where no
    per-channel file exists to point at.
    """
    if isinstance(src, (str, Path)):
        print(f"Memmapping image: {src}")
        img = tifffile.memmap(str(src))
        return img[0] if img.ndim == 3 and img.shape[0] == 1 else img
    return src


def _open_memmaps(dapi_path, morphology_paths):
    """Open DAPI + optional morphology sources lazily; validate shapes.

    Each source may be a path or an opened handle -- see `_as_lazy_source`. Nothing is
    read into memory here: stitched SPRINTseq images are routinely 30k x 30k.

    Returns (dapi_source, morphology_source_list).
    """
    dapi = _as_lazy_source(dapi_path)
    morphs = []
    for p in (morphology_paths or []):
        m = _as_lazy_source(p)
        if tuple(m.shape) != tuple(dapi.shape):
            raise ValueError(
                f"Shape mismatch: morphology {m.shape} vs DAPI {dapi.shape} (source: {p})"
            )
        morphs.append(m)
    return dapi, morphs


def prepare_cellsam_input(dapi_path, morphology_paths=None):
    """
    Build a lazy (H, W, 3) stack for CellSAM, channels drawn from disk-memmapped files.

    CellSAM expects (H, W, 3) with channel 1 = nucleus, channel 2 = cell body. If no
    morphology is supplied, channel 2 stays zero and the model operates on nucleus
    morphology only (useful for nucleus-level workflows).

    Parameters
    ----------
    dapi_path : str | Path
        DAPI (nucleus) image.
    morphology_paths : list[str | Path] | None
        Zero or more cell-body channel images (FAM, CellMask, WGA, etc.). Multiple
        channels are combined via pixel-wise max, computed per-tile at read time.

    Returns
    -------
    _MemmapStack
        Lazy (H, W, 3) stack. Slice it with `stack[y_slice, x_slice]` to materialise
        a single tile; only the slice region is paged in from disk.
    """
    dapi, morphs = _open_memmaps(dapi_path, morphology_paths)
    assignments = [(1, [dapi])]                 # output channel 1 = nucleus
    if morphs:
        assignments.append((2, morphs))         # output channel 2 = cell body (pixel-wise max)
    print(f"Lazy CellSAM stack: shape=(H,W)={dapi.shape} + 3 output channels, "
          f"{len(morphs)} morphology channel{'s' if len(morphs) != 1 else ''} (pixel-wise max'd per tile).")
    return _MemmapStack(reference_shape=dapi.shape, dtype=dapi.dtype,
                        channel_assignments=assignments, n_out_channels=3)


def _apply_cellsam_patches():
    """
    Apply fixes and patches to CellSAM. Ensures patches are applied only once.
    """
    from cellSAM.sam_inference import CellSAM
    import warnings
    
    # 1. Suppress Warnings
    warnings.filterwarnings("ignore", message="Low IOU threshold, ignoring mask.")
    warnings.filterwarnings("ignore", message=".*weights_only=False.*") # Suppress torch.load warning
    warnings.filterwarnings("ignore", message=".*A single label was found in 'y_true' and 'y_pred'.*") # Suppress sklearn confusion matrix warning

    # 2. Monkey Patch CellSAM to fix "not a sequence" crash
    # Instead of patching predict, we wrap segment_cellular_image to catch the specific crash
    import cellSAM.model
    import sys
    
    _original_segment_cellular_image = cellSAM.model.segment_cellular_image

    def _patched_segment_cellular_image(img, model, **kwargs):
        try:
            return _original_segment_cellular_image(img, model, **kwargs)
        except (TypeError, AttributeError) as e:
            # Catch "not a sequence", "iterable", "unpack", OR "NoneType object has no attribute"
            err_str = str(e)
            if any(x in err_str for x in ["not a sequence", "iterable", "unpack", "NoneType", "ndim"]):
                # print(f"DEBUG: Caught empty chunk error: {e}. Returning empty mask.")
                # Return empty result: (mask, embedding, boxes)
                h, w = img.shape[:2]
                return np.zeros((h, w), dtype=np.int32), None, None
            raise e
        except Exception as e:
            # Re-raise other errors
            raise e

    # NUCLEAR OPTION: Iterate over all loaded modules and replace the reference
    count_patched = 0
    for mod_name, mod in list(sys.modules.items()):
        try:
            if hasattr(mod, 'segment_cellular_image'):
                if getattr(mod, 'segment_cellular_image') == _original_segment_cellular_image:
                    setattr(mod, 'segment_cellular_image', _patched_segment_cellular_image)
                    count_patched += 1
        except:
            pass
            
    # Explicitly ensure model definition is updated
    cellSAM.model.segment_cellular_image = _patched_segment_cellular_image

    # 4. Monkey Patch cellSAM.utils.is_low_contrast_clahe to suppress prints
    import cellSAM.utils
    from skimage.exposure import equalize_adapthist
    
    def _patched_is_low_contrast_clahe(image, lower_threshold=0.04, upper_threshold=0.05, kernel_size=256):
        cp = equalize_adapthist(image, kernel_size=kernel_size)
        diff = np.abs(image - cp)
        diff = diff[diff > 0]
        mean_diff = np.median(diff)
        mean_std = np.std(diff)
        # print(f"Mean diff: {mean_diff}")  <-- SILENCED
        # print(np.mean(cp))                <-- SILENCED 
        islowcontrast = lower_threshold < mean_diff < upper_threshold
        return [islowcontrast, mean_diff, mean_std]
    
    cellSAM.utils.is_low_contrast_clahe = _patched_is_low_contrast_clahe

    # 4b. Monkey Patch cellSAM.utils.enhance_low_contrast to drop dead-letter
    # `model.bbox_threshold = ...` lines (upstream bug — `model` is not a
    # parameter of this function and crashes the moment the branch is hit).
    # bbox_threshold is already controlled via run_cellsam_segmentation's
    # kwargs one frame up, so dropping these lines is a pure no-op fix.
    from skimage.exposure import adjust_gamma

    def _patched_enhance_low_contrast(
        img,
        lower_contrast_threshold=0.04,
        upper_contrast_threshold=0.05,
        max_green_channel_value=0,
        clip_limit_default=0.03,
        kernel_size_default=128,
        gamma_default=0.5,
        clip_limit_high_diff=0.05,
        kernel_size_high_diff=64,
        gamma_high_diff=0.7,
        bbox_threshold_high_diff=0.3,
        clip_limit_very_high_diff=0.07,
        bbox_threshold_very_high_diff=0.2,
        clip_limit_adjusted=0.01,
        std_range=(0.005, 0.02),
        mean_diff_threshold=0.07,
        mean_std_threshold=0.05,
    ):
        low_contrast, mean_diff, mean_std = cellSAM.utils.is_low_contrast_clahe(
            img,
            lower_threshold=lower_contrast_threshold,
            upper_threshold=upper_contrast_threshold,
        )
        low_contrast = (
            (low_contrast and img[..., 1].max() == max_green_channel_value)
            if mean_diff < mean_std_threshold else low_contrast
        )
        if low_contrast:
            clip_limit = clip_limit_default
            kernel_size = kernel_size_default
            gamma = gamma_default
            if mean_diff > lower_contrast_threshold and mean_std < mean_std_threshold:
                clip_limit = clip_limit_high_diff
                kernel_size = kernel_size_high_diff
                gamma = gamma_high_diff
            if mean_diff > mean_diff_threshold and mean_std < mean_std_threshold:
                clip_limit = clip_limit_very_high_diff
            if mean_diff > mean_diff_threshold and (std_range[0] < mean_std < std_range[1]):
                clip_limit = clip_limit_adjusted
            img = equalize_adapthist(img, kernel_size=kernel_size, clip_limit=clip_limit)
            img = adjust_gamma(img, gamma=gamma)
        return img

    cellSAM.utils.enhance_low_contrast = _patched_enhance_low_contrast

    # 5. Monkey Patch postprocess_predictions with optimized vectorized version
    from scipy.ndimage import gaussian_filter, find_objects
    from skimage.morphology import disk, binary_opening
    from segment_anything.utils.amg import remove_small_regions
    
    _original_postprocess = cellSAM.model.postprocess_predictions
    
    def _optimized_postprocess_predictions(all_masks: np.ndarray):
        """
        Optimized version of postprocess_predictions with two key improvements:
        
        1. **Memory Optimization**: Avoids creating huge (N_cells, H, W) array.
           Instead, directly assigns to result_mask, avoiding np.max() bottleneck.
        
        2. **Region Cropping**: Only processes bounding box of each cell + padding,
           dramatically reducing computation for sparse cells.
        
        Expected speedup: 10-50x (memory) + 2-5x (cropping) = 20-250x total.
        """
        if all_masks.max() == 0:
            return all_masks
        
        # Get unique mask values (cell IDs)
        mask_values = np.unique(all_masks)
        mask_values = mask_values[mask_values > 0]  # Exclude background
        
        if len(mask_values) == 0:
            return all_masks
        
        # Initialize result mask (same shape as input)
        result_mask = np.zeros_like(all_masks, dtype=all_masks.dtype)
        
        # Add extra padding for gaussian_filter (sigma=3, so ~10 pixels)
        padding = 15  # Safe padding for all operations
        
        # Process each cell with region cropping
        for mask_value in mask_values:
            # Extract binary mask for this cell
            full_mask = (all_masks == mask_value)
            
            if not full_mask.any():
                continue
            
            # Find bounding box of this cell. scipy>=1.15 rejects bool input,
            # so cast the (all_masks == mask_value) mask to a labeled uint8 array.
            objects = find_objects(full_mask.astype(np.uint8))
            if not objects or objects[0] is None:
                continue
            
            y_slice, x_slice = objects[0]
            y_min, y_max = y_slice.start, y_slice.stop
            x_min, x_max = x_slice.start, x_slice.stop
            
            # Add padding, but clip to image boundaries
            h, w = all_masks.shape
            y_min_crop = max(0, y_min - padding)
            y_max_crop = min(h, y_max + padding)
            x_min_crop = max(0, x_min - padding)
            x_max_crop = min(w, x_max + padding)
            
            # Extract region of interest (ROI)
            roi_mask = full_mask[y_min_crop:y_max_crop, x_min_crop:x_max_crop].copy()
            
            # Step 1: Remove small holes and islands (on ROI only)
            roi_mask, _ = remove_small_regions(roi_mask, 20, mode="holes")
            roi_mask, _ = remove_small_regions(roi_mask, 20, mode="islands")
            
            # Step 2-4: Morphological operations (on ROI only, using skimage)
            roi_mask = binary_opening(roi_mask, disk(10))
            
            # Step 5: Gaussian filter (on ROI only)
            roi_mask_float = gaussian_filter(roi_mask.astype(np.float32), sigma=3)
            roi_mask = roi_mask_float > 0.5
            
            # Step 6: Write back to full result_mask
            # Extract the processed region (excluding padding to avoid edge artifacts)
            inner_y_start = y_min - y_min_crop
            inner_y_end = y_max - y_min_crop
            inner_x_start = x_min - x_min_crop
            inner_x_end = x_max - x_min_crop
            
            roi_processed = roi_mask[inner_y_start:inner_y_end, inner_x_start:inner_x_end]
            
            # Convert to same dtype as all_masks and assign cell ID
            # Match original behavior: mask.astype(np.uint8) * mask_value
            processed_with_id = roi_processed.astype(np.uint8) * mask_value
            
            # Assign to result_mask (use max to handle overlaps, matching original np.max logic)
            result_region = result_mask[y_min:y_max, x_min:x_max]
            result_region = np.maximum(result_region, processed_with_id.astype(all_masks.dtype))
            result_mask[y_min:y_max, x_min:x_max] = result_region
        
        return result_mask
    
    cellSAM.model.postprocess_predictions = _optimized_postprocess_predictions

    CellSAM._is_patched_by_utils = True # Mark as patched
    print(f"DEBUG: Applied Monkey Patch to {count_patched} locations (Nuclear Option) + Silenced Utils + Optimized Postprocess.")


def _run_tiled_inference(img, predict_func, block_size=2048, overlap=256):
    """
    Generic function for running tiled inference with center-based object filtering.
    
    Args:
        img: Large image array (H, W, C).
        predict_func: Callback function `f(tile) -> mask` where mask is (tile_h, tile_w).
        block_size: Size of tile.
        overlap: Padding size.
        
    Returns:
        masks_full: Stitched global mask.
    """
    from tqdm import tqdm
    from scipy.ndimage import find_objects
    
    h, w = img.shape[:2]
    masks_full = np.zeros((h, w), dtype=np.uint32)
    max_cell_id = 0
    
    n_rows = int(np.ceil(h / block_size))
    n_cols = int(np.ceil(w / block_size))
    total_tiles = n_rows * n_cols
    
    print(f"Starting Tiled Inference: {n_rows}x{n_cols} = {total_tiles} tiles.")
    print(f"Block size: {block_size}, Padding: {overlap}")
    
    pbar = tqdm(total=total_tiles, desc="Processing Tiles", unit="tile")
    
    for r in range(n_rows):
        for c in range(n_cols):
            # 1. Define regions
            y0, y1 = r * block_size, min((r + 1) * block_size, h)
            x0, x1 = c * block_size, min((c + 1) * block_size, w)
            
            input_y0, input_y1 = max(0, y0 - overlap), min(h, y1 + overlap)
            input_x0, input_x1 = max(0, x0 - overlap), min(w, x1 + overlap)
            
            # 2. Extract tile
            tile = img[input_y0:input_y1, input_x0:input_x1]
            
            # 3. Predict (Callback)
            masks_tile = predict_func(tile)
            
            # 4. Center-based Stitching
            valid_inner_y0 = y0 - input_y0
            valid_inner_y1 = y1 - input_y0
            valid_inner_x0 = x0 - input_x0
            valid_inner_x1 = x1 - input_x0
            
            masks_kept = np.zeros(masks_tile.shape, dtype=np.uint32)
            labels = np.unique(masks_tile)
            labels = labels[labels > 0]
            
            if len(labels) > 0:
                slices = find_objects(masks_tile)
                for label in labels:
                    sl = slices[label-1]
                    if sl is None: continue
                    cy = (sl[0].start + sl[0].stop) / 2
                    cx = (sl[1].start + sl[1].stop) / 2
                    
                    if (valid_inner_y0 <= cy < valid_inner_y1) and \
                       (valid_inner_x0 <= cx < valid_inner_x1):
                        max_cell_id += 1
                        obj_mask = masks_tile[sl] == label
                        masks_kept[sl][obj_mask] = max_cell_id
            
            # 5. Write to Global
            global_slice = (slice(input_y0, input_y1), slice(input_x0, input_x1))
            update_mask = masks_kept > 0
            masks_full[global_slice][update_mask] = masks_kept[update_mask]
            
            pbar.update(1)
            
    pbar.close()
    print(f"Segmentation complete. Total cells: {max_cell_id}")
    return masks_full


def run_cellsam_segmentation(img_combined, block_size=2048, overlap=256, **kwargs):
    """
    Run CellSAM segmentation using MANUAL TILING to reduce memory usage.
    
    Args:
        img_combined: (H, W, 3) array. Channel 1=DAPI, Channel 2=FAM.
        block_size: Tile size for tiling.
        overlap: Overlap between tiles.
        **kwargs: Additional arguments:
            - model: Model name ('cellsam_general', 'cellsam_extra', etc.). Default: None (uses default).
            - bbox_threshold: Bounding box detection threshold. Default: 0.4.
            - postprocess: Whether to apply postprocessing. Default: True (uses optimized version).
            - low_contrast_enhancement: Whether to enhance low contrast. Default: False.
    
    Returns:
        masks: (H, W) array with cell IDs.
    """
    try:
        import torch
        # Ensure patches are applied FIRST (includes optimized postprocess_predictions)
        _apply_cellsam_patches()
        # Import AFTER patching to ensure we get the patched function reference
        from cellSAM.utils import normalize_image
        from cellSAM.model import get_model, segment_cellular_image
    except ImportError as e:
        raise ImportError(
            "CellSAM is required for run_cellsam_segmentation. "
            "cellSAM is not available on PyPI under this name — install the vendored tree: "
            "pip install -e <SPRINTseq>/experiments/src/cellsam"
        ) from e
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"✅ CellSAM is using GPU: {torch.cuda.get_device_name(0)}")
    
    print("Loading CellSAM model...")
    # Support model selection via kwargs
    model_name = kwargs.get('model', None)
    if model_name:
        model = get_model(model=model_name)
    else:
        model = get_model()
    model = model.to(device)
    model.eval()
    
    # Define prediction callback for single tile
    def predict_tile(tile):
        # Convert to float and normalize LOCALLY
        tile_float = tile.astype(np.float32)
        tile_float = normalize_image(tile_float)
        
        # Args mapping
        bbox_thresh = kwargs.get('bbox_threshold', 0.4)
        postprocess = kwargs.get('postprocess', True)  # Default: True (uses optimized version)
        
        if kwargs.get('low_contrast_enhancement', False):
            from cellSAM.utils import enhance_low_contrast
            tile_float = enhance_low_contrast(tile_float)
        
        # Run inference
        masks, _, _ = segment_cellular_image(
            tile_float, 
            model=model, 
            normalize=False, # Already normalized above
            device=device,
            bbox_threshold=bbox_thresh,
            postprocess=postprocess  # Uses optimized version if True
        )
        return masks

    # Run generic tiling
    return _run_tiled_inference(img_combined, predict_tile, block_size, overlap)


def prepare_cellpose_input(dapi_path, morphology_paths=None):
    """
    Build a lazy (H, W, 2) stack for Cellpose, channels drawn from disk-memmapped files.

    Cellpose expects (H, W, 2) with channel 0 = cytoplasm, channel 1 = nucleus.
    If no morphology is supplied, channel 0 stays zero and Cellpose falls back to
    nucleus-only segmentation.

    Parameters
    ----------
    dapi_path : str | Path
        DAPI (nucleus) image.
    morphology_paths : list[str | Path] | None
        Zero or more cytoplasm / membrane channels; combined via pixel-wise max
        per tile at read time.

    Returns
    -------
    _MemmapStack
        Lazy (H, W, 2) stack. Only the sliced region is paged in from disk per tile.
    """
    dapi, morphs = _open_memmaps(dapi_path, morphology_paths)
    assignments = [(1, [dapi])]                 # output channel 1 = nucleus
    if morphs:
        assignments.append((0, morphs))         # output channel 0 = cytoplasm (pixel-wise max)
    print(f"Lazy Cellpose stack: shape=(H,W)={dapi.shape} + 2 output channels, "
          f"{len(morphs)} morphology channel{'s' if len(morphs) != 1 else ''} (pixel-wise max'd per tile).")
    return _MemmapStack(reference_shape=dapi.shape, dtype=dapi.dtype,
                        channel_assignments=assignments, n_out_channels=2)


def run_cellpose_segmentation(img_stacked, pretrained_model='cpsam', use_gpu=True):
    """
    Run Cellpose segmentation on prepared image using MANUAL TILING.
    """
    try:
        from cellpose import models
    except ImportError as e:
        raise ImportError(
            "cellpose is required for run_cellpose_segmentation. "
            "Install via PyPI: pip install cellpose  "
            "(or from the vendored tree: pip install -e <SPRINTseq>/experiments/src/cellpose)."
        ) from e
    
    print(f"Initializing Cellpose model: {pretrained_model} (GPU={use_gpu})")
    # Cellpose v4 API change: 'Cellpose' class removed, use 'CellposeModel'
    # 'model_type' argument deprecated, use 'pretrained_model'
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=pretrained_model)
    
    print("Skipping diameter estimation for CellposeModel/cpsam (using default)...")
    diam_est = None
    
    # Define prediction callback
    def predict_tile(tile):
        # Returns: masks, flows, styles (3 values)
        # channels argument is deprecated in v4 and triggers warning. 
        # Input tile is (H, W, 2) [Cyto, Nuclei], which Cellpose handles automatically.
        masks, _, _ = model.eval(tile, diameter=diam_est)
        return masks
        
    # Run generic tiling (Default block_size=2048, overlap=128 from previous config)
    return _run_tiled_inference(img_stacked, predict_tile, block_size=2048, overlap=128)


def assign_spots_to_cells(spots_df, segmentation_mask):
    """
    Assign RNA spots to cells based on the segmentation mask.
    """
    print("Assigning spots to cells...")
    
    h, w = segmentation_mask.shape
    
    # Create valid mask for spots within image boundaries
    valid_spots = (
        (spots_df['X'] >= 0) & 
        (spots_df['X'] < w) & 
        (spots_df['Y'] >= 0) & 
        (spots_df['Y'] < h)
    )
    
    n_dropped = (~valid_spots).sum()
    if n_dropped > 0:
        print(f"Warning: {n_dropped} spots are outside the segmentation mask boundaries and will be ignored.")
        
    valid_spots_df = spots_df[valid_spots].copy()
    
    y_coords = valid_spots_df['Y'].astype(int).values
    x_coords = valid_spots_df['X'].astype(int).values
    
    cell_ids = segmentation_mask[y_coords, x_coords]
    
    valid_spots_df['Cell_ID'] = cell_ids
    
    # Filter out spots that landed on background (Cell_ID == 0)
    assigned_spots = valid_spots_df[valid_spots_df['Cell_ID'] > 0].copy()
    print(f"Assigned {len(assigned_spots)} spots to {assigned_spots['Cell_ID'].nunique()} unique cells.")
    
    print("Generating expression matrix...")
    expression_matrix = pd.crosstab(assigned_spots['Cell_ID'], assigned_spots['Gene'])
    
    return assigned_spots, expression_matrix


def extract_cell_positions(segmentation_mask):
    """
    Extract centroid positions and areas for all cells in the segmentation mask.
    Uses vectorized operations to avoid slow for loops.
    
    Args:
        segmentation_mask: (H, W) array with cell IDs. Background = 0.
    
    Returns:
        DataFrame with columns: Cell_ID, X, Y, Area
            - Cell_ID: Cell identifier
            - X: Centroid X coordinate (in pixels)
            - Y: Centroid Y coordinate (in pixels)
            - Area: Cell area in pixels
    """
    from skimage.measure import regionprops_table
    
    print("Extracting cell positions...")
    
    # Get unique cell IDs (exclude background = 0)
    cell_ids = np.unique(segmentation_mask)
    cell_ids = cell_ids[cell_ids > 0]
    
    if len(cell_ids) == 0:
        print("Warning: No cells found in segmentation mask.")
        return pd.DataFrame(columns=['Cell_ID', 'X', 'Y', 'Area'])
    
    # Vectorized extraction: use regionprops directly on original mask
    # regionprops can handle non-consecutive labels, so no need to create new array
    from skimage.measure import regionprops
    
    # Extract all properties at once (vectorized, processes entire mask in one pass)
    props = regionprops(segmentation_mask)
    
    # Use list comprehension for faster iteration (faster than explicit for loop)
    # Filter and extract properties in one pass
    cell_ids_set = set(cell_ids)  # For fast O(1) lookup
    
    # List comprehension is faster than explicit loop + append
    positions = [
        {
            'Cell_ID': int(prop.label),
            'Y': float(prop.centroid[0]),  # Row = Y
            'X': float(prop.centroid[1]),  # Column = X
            'Area': int(prop.area)
        }
        for prop in props
        if prop.label in cell_ids_set
    ]
    
    # Convert to DataFrame
    positions_df = pd.DataFrame(positions)
    positions_df = positions_df.sort_values('Cell_ID').reset_index(drop=True)

    print(f"Extracted positions for {len(positions_df)} cells.")

    return positions_df


def assign_spots_to_nuclei_kdtree(spots_df, centroids_df, max_distance=None):
    """
    Assign RNA spots to the nearest nucleus centroid via a KD-tree.

    Use this when no morphology channel is available — segment nuclei from DAPI only,
    take their centroids, and attribute each spot to its nearest nucleus. An optional
    distance cap (in pixels) drops spots that fall too far from any nucleus (likely
    extracellular noise).

    Parameters
    ----------
    spots_df : pd.DataFrame
        Must contain columns 'Y', 'X', 'Gene'.
    centroids_df : pd.DataFrame
        Nucleus centroids with columns 'Cell_ID', 'Y', 'X' (as produced by extract_cell_positions).
    max_distance : float | None
        Pixel radius beyond which a spot is considered unassigned (dropped). None = no cap.

    Returns
    -------
    assigned_spots : pd.DataFrame
        spots_df subset augmented with 'Cell_ID' and 'Distance' (pixels to assigned nucleus).
    expression_matrix : pd.DataFrame
        Cell x Gene count matrix.
    """
    from scipy.spatial import cKDTree

    if centroids_df.empty:
        print("Warning: no nucleus centroids supplied; returning empty assignment.")
        return spots_df.iloc[0:0].copy().assign(Cell_ID=pd.Series(dtype=int),
                                                 Distance=pd.Series(dtype=float)), \
               pd.DataFrame()

    centroid_yx = centroids_df[['Y', 'X']].to_numpy()
    tree = cKDTree(centroid_yx)

    spot_yx = spots_df[['Y', 'X']].to_numpy()
    distances, indices = tree.query(spot_yx, k=1)
    cell_ids = centroids_df['Cell_ID'].to_numpy()[indices]

    out = spots_df.copy()
    out['Cell_ID'] = cell_ids
    out['Distance'] = distances

    if max_distance is not None:
        n_before = len(out)
        out = out[out['Distance'] <= max_distance].copy()
        print(f"Dropped {n_before - len(out)} spots beyond {max_distance} px from any nucleus.")

    print(f"Assigned {len(out)} spots to {out['Cell_ID'].nunique()} unique nuclei (KD-tree).")

    print("Generating expression matrix...")
    expression_matrix = pd.crosstab(out['Cell_ID'], out['Gene'])
    return out, expression_matrix


def auto_detect_dapi(stitched_dir, preferred_cycles=(11, 1)):
    """Locate the DAPI mosaic, preferring the given cycles.

    Returns a **path** for the legacy per-file layout and an **opened lazy handle** for a
    run stored as one OME-Zarr store -- both are accepted downstream by
    `prepare_cellsam_input` / `prepare_cellpose_input`. Raises FileNotFoundError if there
    is no DAPI mosaic at all.
    """
    stc = Path(stitched_dir)
    if mosaic.backend(stc) == "zarr":
        available = [c for c, ch in mosaic.list_mosaics(stc) if ch == "DAPI"]
        if not available:
            raise FileNotFoundError(f"No DAPI mosaic in the store under {stc}")
        for cyc in preferred_cycles:
            if cyc in available:
                return mosaic.open_mosaic(stc, cyc, "DAPI")
        return mosaic.open_mosaic(stc, sorted(available)[0], "DAPI")

    for cyc in preferred_cycles:
        p = stc / f"cyc_{cyc}_DAPI.tif"
        if p.exists():
            return p
    any_dapi = sorted(stc.glob("cyc_*_DAPI.tif"))
    if any_dapi:
        return any_dapi[0]
    raise FileNotFoundError(f"No cyc_*_DAPI.tif found under {stc}")


def auto_detect_morphology(stitched_dir, names=("FAM",)):
    """Locate one morphology mosaic per requested channel name.

    Same dual return as `auto_detect_dapi`: paths on the legacy layout, opened handles on
    a converted run. Empty list if none found -- the caller then switches to
    nuclei-kdtree mode.
    """
    stc = Path(stitched_dir)
    found = []
    if mosaic.backend(stc) == "zarr":
        for name in names:
            cycles = [c for c, ch in mosaic.list_mosaics(stc) if ch == name]
            if cycles:
                found.append(mosaic.open_mosaic(stc, sorted(cycles)[0], name))
        return found

    for name in names:
        matches = sorted(stc.glob(f"cyc_*_{name}.tif"))
        if matches:
            found.append(matches[0])
    return found
