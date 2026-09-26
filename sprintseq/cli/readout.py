"""
Stitched-image readout pipeline for SPRINTseq.

This pipeline performs spot detection and intensity readout directly from
stitched images using memory-mapped block-based processing. Since stitched
images are already in global coordinates, no coordinate transform is needed.

Pipeline stages:
1. Spot detection using spotiflow (or traditional methods) on stitched images
2. Intensity extraction using tophat+local-max method
3. Conservative intensity filtering
4. Deduplication across block overlaps
5. Output position.csv and intensity.csv

Usage:
    sprintseq readout --run-id <run_id>
"""

import argparse
import logging
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

from sprintseq.readout import (
    get_spot_coordinates,
    read_intensity_tophat,
    block_starts,
)
from sprintseq.readout.deduplicate import deduplicate_dataframe
from sprintseq.readout.mosaic import has_mosaic, mosaic_shape, open_mosaic
from sprintseq.readout.spot_detection import mosaic_percentiles

# Logging
_LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
logger = logging.getLogger(__name__)

# ========== Configuration ==========
DEFAULT_RUN_ID = 'example_dataset'

BASE_DEST_DIRECTORY = r'\\10.10.10.1\NAS Processed Images'
CHANNELS = ['cy3', 'cy5']                    # SBS spot channels (both detection and intensity)

# Spot detection — pass an explicit list of cycles to detect in. Typical choices:
#   [1, 2, 3, 4]   classic: detect on first 4 sequencing cycles, union across channels
#   [11]           total-spot: single dedicated cycle where every spot is labelled
#   [1, 2, 3, 4, 11]  both strategies combined for maximum recall
DETECTION_CYCLES = [1, 2, 3, 4]
SNRS = {'cy3': 3.0, 'cy5': 3.0}
DETECTION_METHOD = 'spotiflow'

# Spotiflow defaults. hybiss recovers ~+90 % weak cy3 spots vs `general`
# (established run 6 iter6, 2026-04-25, BZ29 TNBC marker panel). prob_thresh
# is intentionally very permissive — postcode downstream filters noise via
# Background/Infeasible classes. hybiss + prob_thresh=0.01 is the standing
# readout default (set 2026-05-30). Re-tune per panel using
# experiments/notebooks/readout_spotiflow_test.ipynb before each new run and
# overwrite these two values; they are the single source of truth for the
# pipeline's Spotiflow config.
SPOTIFLOW_PRETRAINED_NAME = 'hybiss'
SPOTIFLOW_PROB_THRESH = 0.01
# How Spotiflow's input is scaled to [0, 1] before prediction:
#   'block'  -- Spotiflow's own default: each 2048^2 block by its own p1/p99.8, so a
#               block's contrast depends on what else is in it.
#   'global' -- one p1/p99.8 per mosaic (cycle x channel), estimated by
#               spot_detection.mosaic_percentiles and shared by all its blocks.
SPOTIFLOW_NORMALIZATIONS = ('block', 'global')
SPOTIFLOW_NORMALIZATION = 'block'

# Intensity readout — sequencing cycles are consecutive 1..SEQ_CYCLE
SEQ_CYCLE = 10
TOPHAT_RADIUS = 3
SEARCH_RADIUS = 1
MIN_INTENSITY_THRESHOLD = 50

# Block processing
BLOCK_SIZE = (2048, 2048)
BLOCK_OVERLAP = (64, 64)

# Deduplication
DEDUPLICATE_THRESHOLD = 2  # Pixels (Chebyshev distance)

# Parallelism
N_WORKERS = 4

# Methods accepted by `--detection-method` and dispatched in
# sprintseq.readout.spot_detection.get_spot_coordinates.
DETECTION_METHODS = (
    'spotiflow',
    'blob_log',
    'dog', 'gaussian_dog',
    'tophat', 'gaussian_tophat',
)


def _default_detection_kwargs(detection_method, channel, snrs=None):
    """Default per-method kwargs forwarded to get_spot_coordinates.

    Centralises method-specific defaults so the same dict drives both the
    in-process pipeline and any external test or per-run override script.
    Only the default values live here; the user is free to fork and tune.

    Parameters
    ----------
    detection_method : str
        One of DETECTION_METHODS.
    channel : str
        SBS channel name (e.g. 'cy3'); used to look up per-channel SNR for
        the traditional / classical detection methods.
    snrs : dict, optional
        Mapping of channel -> SNR threshold. Falls back to module-level SNRS.

    Returns
    -------
    dict
        Keyword args to pass to get_spot_coordinates(image, method=..., **kwargs).
    """
    if detection_method == 'spotiflow':
        return {
            'prob_thresh': SPOTIFLOW_PROB_THRESH,
            'device': None,
            'pretrained_name': SPOTIFLOW_PRETRAINED_NAME,
        }
    if detection_method == 'blob_log':
        # Approximate match to Fiji TrackMate LogDetector with radius=2.0.
        # sigma ~ radius/sqrt(2) ~ 1.41; threshold lives on skimage's
        # img_as_float + scale-normalized LoG response, retune empirically.
        return {
            'min_sigma': 1.0, 'max_sigma': 2.0, 'num_sigma': 2,
            'threshold': 0.005, 'overlap': 0.5,
        }
    snrs = snrs if snrs is not None else SNRS
    kwargs = {'snr': snrs.get(channel, 3.0)}
    # `snr` is popped by get_spot_coordinates and used for the threshold; everything else
    # is forwarded to the feature extractor, so it must match THAT function's signature.
    # Only the tophat family takes a radius -- feature_dog / feature_gaussian_dog take
    # (sigma1, sigma2, normalize_percentile) and used to raise TypeError on this.
    if 'tophat' in detection_method:
        kwargs['tophat_radius'] = TOPHAT_RADIUS
    return kwargs


# ========== Worker Functions (module-level for pickling) ==========

def _detect_spots_in_block(args):
    """Worker: detect spots in one block. Returns (channel, global_coords, error).

    Receives block as plain numpy array (not memmap) for cross-process pickling.
    """
    channel, block, start_y, start_x, method, method_kwargs = args
    try:
        coords_local = get_spot_coordinates(block, method=method, min_distance=2, **method_kwargs)
        if len(coords_local) == 0:
            return (channel, np.empty((0, 2), dtype=np.float64), None)
        global_coords = coords_local + np.array([start_y, start_x], dtype=coords_local.dtype)
        return (channel, global_coords, None)
    except Exception as e:
        return (channel, np.empty((0, 2), dtype=np.float64), str(e))


def _read_intensity_in_block(args):
    """Worker: read intensity for spots in one block. Returns (col_name, indices, intensities).

    Receives block as plain numpy array (not memmap) for cross-process pickling.
    """
    col_name, block, indices, coords_local, tophat_radius, search_radius = args
    if len(indices) == 0:
        return (col_name, np.array([], dtype=np.int64), np.array([], dtype=np.float64), None)
    try:
        intensities = read_intensity_tophat(
            block, coords_local,
            tophat_radius=tophat_radius, search_radius=search_radius
        )
        return (col_name, indices, intensities, None)
    except Exception as e:
        return (col_name, indices, np.full(len(indices), np.nan, dtype=np.float64), str(e))


# ========== Stage Functions ==========

def remap_mosaic(cyc, channel):
    """Logical ``(cycle, channel)`` -> physical ``(cycle, channel)``. Identity by default.

    Override this when a run's logical cycles do not line up with what was acquired --
    e.g. VIS5c HALF, where logical cyc 2 is physically cyc 3 because a DAPI-activation
    scan sits in between::

        LOGICAL_TO_PHYSICAL = {1: 1, 2: 3}
        ro.remap_mosaic = lambda c, ch: (LOGICAL_TO_PHYSICAL.get(c, c), ch)

    This is the single indirection point, and it works for both storage backends. Patching
    `_get_stitched_image_path` instead only ever affected the legacy per-file layout, and
    silently does nothing for a run stored as one OME-Zarr store.
    """
    return cyc, channel


def _get_stitched_image_path(stc_dir, cyc, channel):
    """Physical path of a stitched mosaic in the legacy per-file layout.

    Applies `remap_mosaic`. Only meaningful when the run is stored as individual TIFFs --
    use `_has_stitched` / `_open_stitched` for anything that must also work on a converted
    run.
    """
    cyc, channel = remap_mosaic(cyc, channel)
    return Path(stc_dir) / f'cyc_{cyc}_{channel}.tif'


def _has_stitched(stc_dir, cyc, channel):
    """Backend-agnostic existence check, honouring `remap_mosaic`."""
    return has_mosaic(stc_dir, *remap_mosaic(cyc, channel))


def _open_stitched(stc_dir, cyc, channel):
    """Open a stitched mosaic for block slicing, honouring `remap_mosaic`.

    Returns something indexable as ``[y0:y1, x0:x1]``; reads stay lazy on both backends,
    so the block loops behave as they did against ``tifffile.memmap``.
    """
    return open_mosaic(stc_dir, *remap_mosaic(cyc, channel))


def _stitched_shape(stc_dir, cyc, channel):
    """Mosaic ``(h, w)`` without reading pixels, honouring `remap_mosaic`."""
    return mosaic_shape(stc_dir, *remap_mosaic(cyc, channel))


def _filter_coords_by_coverage(coords, mask, downsample=16):
    """Keep only (Y, X) coords that fall inside a boolean coverage mask.

    Generic gate for multi-cycle co-decoding: when cycles are imaged over different
    footprints (e.g. cyc1 brain-only, cyc2 full-slide), spots outside the cycle-coverage
    intersection have missing-cycle intensities ~0 and decode to false codes. `mask` is a
    2-D boolean array downsampled by `downsample`; keep = mask[Y//d, X//d] (edge-clipped).
    `mask=None` (or empty coords) is a no-op. Returns (kept_coords, keep_bool).
    """
    keep = np.ones(len(coords), dtype=bool)
    if mask is None or len(coords) == 0:
        return coords, keep
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    yi = np.clip(coords[:, 0].astype(np.int64) // downsample, 0, h - 1)
    xi = np.clip(coords[:, 1].astype(np.int64) // downsample, 0, w - 1)
    keep = mask[yi, xi]
    return coords[keep], keep


def detect_all_spots(
    stc_dir,
    channels=CHANNELS,
    detection_cycles=None,
    block_size=BLOCK_SIZE,
    block_overlap=BLOCK_OVERLAP,
    detection_method=DETECTION_METHOD,
    snrs=SNRS,
    n_workers=N_WORKERS,
    coverage_mask=None,
    coverage_downsample=16,
    spotiflow_normalization=None,
):
    """Stage 1: Detect spots across detection cycles/channels from stitched images.

    For each detection image (cycle x channel from detection_cycles x channels):
      - Open tifffile.memmap
      - Generate blocks via block_starts()
      - Submit blocks to ProcessPoolExecutor
      - Accumulate global coordinates
    Combine all coordinates, remove exact duplicates.

    Parameters
    ----------
    detection_cycles : list[int] | None
        Explicit list of cycle numbers to detect in. Defaults to `DETECTION_CYCLES`
        (= [1, 2, 3, 4]). Use e.g. `[11]` when your protocol stains every spot in a
        single dedicated cycle, or `[1, 2, 3, 4, 11]` to union both strategies.
    coverage_mask : np.ndarray | None
        Optional 2-D boolean coverage mask (downsampled by `coverage_downsample`). When
        given, detected coordinates outside the imaged-region intersection are dropped
        (keep = mask[Y//d, X//d]). For multi-cycle co-decoding where cycles span different
        footprints (e.g. VS200 cyc1 brain-only vs cyc2 full-slide); see
        `_filter_coords_by_coverage`. Default None = no gating (standard SBS behavior).
    spotiflow_normalization : {'block', 'global'} | None
        Spotiflow input scaling, see `SPOTIFLOW_NORMALIZATION` (the default when None).
        Ignored by the other detection methods. With 'global', each mosaic's (mi, ma)
        is logged and returned in ``detection_stats['norm_range']``.

    Returns
    -------
    np.ndarray
        (N, 2) array of unique global (Y, X) coordinates.
    """
    stc_dir = Path(stc_dir)
    by, bx = block_size
    if detection_cycles is None:
        detection_cycles = DETECTION_CYCLES
    if spotiflow_normalization is None:
        spotiflow_normalization = SPOTIFLOW_NORMALIZATION
    if spotiflow_normalization not in SPOTIFLOW_NORMALIZATIONS:
        raise ValueError(f"spotiflow_normalization={spotiflow_normalization!r}; "
                         f"expected one of {SPOTIFLOW_NORMALIZATIONS}")
    global_norm = detection_method == 'spotiflow' and spotiflow_normalization == 'global'
    norm_ranges = {}

    # Count total blocks for progress bar
    n_total_blocks = 0
    for cyc in detection_cycles:
        for channel in channels:
            if not _has_stitched(stc_dir, cyc, channel):
                logger.warning(f"Stitched mosaic not found: cyc_{cyc}_{channel}, skipping")
                continue
            h, w = _stitched_shape(stc_dir, cyc, channel)
            n_total_blocks += len(block_starts(h, w, block_size, block_overlap))

    # Lazy iterator: open memmap per image, yield blocks as numpy arrays
    def _detection_task_iter():
        for cyc in detection_cycles:
            for channel in channels:
                if not _has_stitched(stc_dir, cyc, channel):
                    continue
                detection_kwargs = _default_detection_kwargs(detection_method, channel, snrs)
                img = _open_stitched(stc_dir, cyc, channel)
                h, w = img.shape
                if global_norm:
                    mi, ma = mosaic_percentiles(img)
                    norm_ranges[f"cyc_{cyc}_{channel}"] = [mi, ma]
                    logger.info(f"  cyc_{cyc}_{channel}: global normalization range "
                                f"[{mi:.1f}, {ma:.1f}]")
                    detection_kwargs = {**detection_kwargs, 'norm_range': (mi, ma)}
                for start_y, start_x in block_starts(h, w, block_size, block_overlap):
                    end_y = min(start_y + by, h)
                    end_x = min(start_x + bx, w)
                    block = np.asarray(img[start_y:end_y, start_x:end_x])
                    yield (channel, block, start_y, start_x, detection_method, detection_kwargs)
                del img

    # Process with sliding window of n_workers tasks
    task_iter = _detection_task_iter()
    results = []
    n_blocks_done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for _ in range(n_workers):
            task = next(task_iter, None)
            if task is None:
                break
            futures[executor.submit(_detect_spots_in_block, task)] = None

        pbar = tqdm(desc='Detecting spots', total=n_total_blocks)
        while futures:
            for future in as_completed(futures):
                n_blocks_done += 1
                pbar.update(1)
                channel, coords, error = future.result()
                if error:
                    logger.warning(f"Detection error in block: {error}")
                elif len(coords) > 0:
                    results.append((channel, coords))
                del futures[future]
                task = next(task_iter, None)
                if task is not None:
                    futures[executor.submit(_detect_spots_in_block, task)] = None
                break
        pbar.close()

    if not results:
        logger.warning("No spots detected in any channel!")
        detection_stats = {"per_channel": {}, "total_raw": 0, "total_after_exact_dedup": 0}
        if global_norm:
            detection_stats["norm_range"] = norm_ranges
        return np.empty((0, 2), dtype=np.float64), detection_stats

    # Log per-channel counts (before exact-duplicate removal)
    per_channel_counts = {}
    coords_by_channel = defaultdict(list)
    for channel, coords in results:
        coords_by_channel[channel].append(coords)
    for channel in channels:
        if channel in coords_by_channel:
            ch_coords = np.vstack(coords_by_channel[channel])
            per_channel_counts[channel] = len(ch_coords)
            logger.info(f"  {channel}: {len(ch_coords)} spots detected (raw, before dedup)")

    # Combine and remove exact duplicates
    all_coords = np.vstack([coords for _, coords in results])
    logger.info(f"Total coordinates before deduplication: {len(all_coords)}")
    coords_rounded = np.round(all_coords).astype(np.int32)
    _, unique_indices = np.unique(coords_rounded, axis=0, return_index=True)
    unique_coords = all_coords[unique_indices]
    logger.info(f"Total coordinates after removing exact duplicates: {len(unique_coords)}")

    detection_stats = {
        "per_channel": per_channel_counts,
        "total_raw": len(all_coords),
        "total_after_exact_dedup": len(unique_coords),
    }
    if detection_method == 'spotiflow':
        detection_stats["spotiflow_normalization"] = spotiflow_normalization
    if global_norm:
        detection_stats["norm_range"] = norm_ranges

    if coverage_mask is not None:
        n_before = len(unique_coords)
        unique_coords, _keep = _filter_coords_by_coverage(
            unique_coords, coverage_mask, coverage_downsample)
        detection_stats["coverage_kept"] = len(unique_coords)
        detection_stats["coverage_dropped"] = n_before - len(unique_coords)
        logger.info(
            f"Coverage gate: {n_before} -> {len(unique_coords)} "
            f"({detection_stats['coverage_dropped']} dropped outside cycle-coverage intersection)"
        )

    return unique_coords, detection_stats


def read_all_intensities(
    stc_dir,
    unique_coords,
    channels=CHANNELS,
    seq_cycle=SEQ_CYCLE,
    block_size=BLOCK_SIZE,
    block_overlap=BLOCK_OVERLAP,
    tophat_radius=TOPHAT_RADIUS,
    search_radius=SEARCH_RADIUS,
    n_workers=N_WORKERS,
):
    """Stage 2: Read intensities for all detected spots from stitched images.

    Precompute which coordinates fall in which block (once for all images).
    For each intensity image (cyc_1..seq_cycle x channels):
      - Open tifffile.memmap
      - For each block containing spots, read with margin and extract intensities

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Y', 'X', 'cyc_1_cy3', 'cyc_1_cy5', ..., 'cyc_N_chM']
    """
    stc_dir = Path(stc_dir)
    by, bx = block_size
    oy, ox = block_overlap
    step_y = max(1, by - oy)
    step_x = max(1, bx - ox)
    margin = tophat_radius + search_radius + 2

    # Initialize DataFrame
    intensity_df = pd.DataFrame({'Y': unique_coords[:, 0], 'X': unique_coords[:, 1]})
    col_names = [f'cyc_{cyc}_{ch}' for cyc in range(1, seq_cycle + 1) for ch in channels]
    for col in col_names:
        intensity_df[col] = np.nan

    # Pre-bucket coordinates into blocks (done once, reused for all images)
    bucket = defaultdict(list)
    for i in range(len(unique_coords)):
        y, x = float(unique_coords[i, 0]), float(unique_coords[i, 1])
        start_y = step_y * (int(y) // step_y)
        start_x = step_x * (int(x) // step_x)
        bucket[(start_y, start_x)].append(i)
    coords_in_block = {key: np.array(idx_list, dtype=np.int64) for key, idx_list in bucket.items()}

    # Count total intensity tasks (blocks with spots x existing images)
    n_existing_images = sum(
        1 for cyc in range(1, seq_cycle + 1) for ch in channels
        if _has_stitched(stc_dir, cyc, ch)
    )
    n_intensity_total = len(coords_in_block) * n_existing_images

    # Lazy iterator: open memmap per image, yield blocks with spots
    def _intensity_task_iter():
        for cyc in range(1, seq_cycle + 1):
            for channel in channels:
                col_name = f'cyc_{cyc}_{channel}'
                if not _has_stitched(stc_dir, cyc, channel):
                    logger.warning(
                        f"Stitched mosaic not found: cyc_{cyc}_{channel}, filling with NaN")
                    continue
                img = _open_stitched(stc_dir, cyc, channel)
                h, w = img.shape
                for start_y, start_x in block_starts(h, w, block_size, block_overlap):
                    indices = coords_in_block.get((start_y, start_x), None)
                    if indices is None or len(indices) == 0:
                        continue
                    # Read block with margin for tophat kernel effects
                    y0 = max(0, start_y - margin)
                    y1 = min(h, min(start_y + by, h) + margin)
                    x0 = max(0, start_x - margin)
                    x1 = min(w, min(start_x + bx, w) + margin)
                    block = np.asarray(img[y0:y1, x0:x1])
                    # Convert global coords to block-local coords
                    coords_global = unique_coords[indices]
                    coords_local = coords_global - np.array([y0, x0], dtype=coords_global.dtype)
                    yield (col_name, block, indices, coords_local, tophat_radius, search_radius)
                del img

    # Process with sliding window
    intensity_task_iter = _intensity_task_iter()
    n_blocks_done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for _ in range(n_workers):
            task = next(intensity_task_iter, None)
            if task is None:
                break
            futures[executor.submit(_read_intensity_in_block, task)] = None

        pbar = tqdm(desc='Reading intensities', total=n_intensity_total)
        while futures:
            for future in as_completed(futures):
                n_blocks_done += 1
                pbar.update(1)
                col_name, indices, intensities, error = future.result()
                if error:
                    logger.warning(f"Intensity read error for {col_name}: {error}")
                if len(indices) > 0:
                    intensity_df.loc[indices, col_name] = intensities
                del futures[future]
                task = next(intensity_task_iter, None)
                if task is not None:
                    futures[executor.submit(_read_intensity_in_block, task)] = None
                break
        pbar.close()

    # Fill any remaining NaN with 0
    for col in col_names:
        if intensity_df[col].isna().any():
            n_miss = intensity_df[col].isna().sum()
            logger.warning(f"  {col}: {n_miss} spots had no intensity (fill with 0)")
            intensity_df[col] = intensity_df[col].fillna(0)

    logger.info(f"Intensity reading completed for {len(intensity_df)} spots")
    return intensity_df


def run_pipeline(run_id, n_workers=None, detection_cycles=None, seq_cycle=None,
                 channels=None, detection_method=None,
                 coverage_mask=None, coverage_downsample=16,
                 spotiflow_normalization=None):
    """Main entry point for stitched-image readout pipeline.

    Parameters
    ----------
    run_id : str
        Run identifier (e.g., '20251128_ZCH_BZ09_Re2_mut_new').
    n_workers : int, optional
        Number of parallel workers. Defaults to N_WORKERS.
    detection_cycles : list[int], optional
        Cycles to run spot detection on. Defaults to DETECTION_CYCLES (=[1,2,3,4]).
        Pass a single-element list (e.g. [11]) if your protocol labels every spot
        in one dedicated cycle, or combine strategies by union (e.g. [1,2,3,4,11]).
    seq_cycle : int, optional
        Number of sequencing cycles (always consecutive 1..seq_cycle) used for
        intensity readout. Defaults to SEQ_CYCLE.
    channels : list[str], optional
        SBS spot channels. Defaults to CHANNELS (['cy3','cy5']). Used for both
        detection and intensity readout.
    detection_method : str, optional
        Detection method dispatched in get_spot_coordinates. One of
        DETECTION_METHODS. Defaults to module-level DETECTION_METHOD.
    spotiflow_normalization : {'block', 'global'}, optional
        Spotiflow input scaling. Defaults to SPOTIFLOW_NORMALIZATION.
    """
    # Copy list defaults so callers can't mutate the module-level state via
    # the returned reference.
    n_workers = N_WORKERS if n_workers is None else n_workers
    detection_cycles = list(DETECTION_CYCLES) if detection_cycles is None else list(detection_cycles)
    seq_cycle = SEQ_CYCLE if seq_cycle is None else seq_cycle
    channels = list(CHANNELS) if channels is None else list(channels)
    detection_method = DETECTION_METHOD if detection_method is None else detection_method
    spotiflow_normalization = (SPOTIFLOW_NORMALIZATION if spotiflow_normalization is None
                               else spotiflow_normalization)
    if detection_method not in DETECTION_METHODS:
        raise ValueError(
            f"Unknown detection_method={detection_method!r}; "
            f"expected one of {DETECTION_METHODS}"
        )

    dest_dir = Path(BASE_DEST_DIRECTORY) / f'{run_id}_processed'
    stc_dir = dest_dir / 'stitched'
    read_dir = dest_dir / 'readout'

    if not stc_dir.exists():
        raise FileNotFoundError(f"Stitched directory not found: {stc_dir}")

    read_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("SPRINTseq Stitched-Image Readout Pipeline")
    logger.info("=" * 80)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Stitched dir: {stc_dir}")
    logger.info(f"Output dir: {read_dir}")
    logger.info(f"Detection: {detection_method}, cycles {detection_cycles}, channels {channels}")
    if detection_method == 'spotiflow':
        logger.info(f"Spotiflow: {SPOTIFLOW_PRETRAINED_NAME}, prob_thresh={SPOTIFLOW_PROB_THRESH}, "
                    f"normalization={spotiflow_normalization}")
    logger.info(f"Intensity: tophat (radius={TOPHAT_RADIUS}, search={SEARCH_RADIUS}), "
                f"cycles 1-{seq_cycle}, channels {channels}")
    logger.info(f"Block size: {BLOCK_SIZE}, overlap: {BLOCK_OVERLAP}")
    logger.info(f"Workers: {n_workers}")
    logger.info("=" * 80)

    # Stage 1: Spot Detection
    logger.info("=" * 80)
    logger.info("Stage 1: Spot Detection")
    logger.info("=" * 80)
    unique_coords, detection_stats = detect_all_spots(
        stc_dir,
        channels=channels,
        detection_cycles=detection_cycles,
        block_size=BLOCK_SIZE,
        block_overlap=BLOCK_OVERLAP,
        detection_method=detection_method,
        snrs=SNRS,
        n_workers=n_workers,
        coverage_mask=coverage_mask,
        coverage_downsample=coverage_downsample,
        spotiflow_normalization=spotiflow_normalization,
    )
    if len(unique_coords) == 0:
        logger.warning("No spots detected. Exiting.")
        return

    # Stage 2: Intensity Reading
    logger.info("=" * 80)
    logger.info("Stage 2: Intensity Reading")
    logger.info("=" * 80)
    intensity_df = read_all_intensities(
        stc_dir,
        unique_coords,
        channels=channels,
        seq_cycle=seq_cycle,
        block_size=BLOCK_SIZE,
        block_overlap=BLOCK_OVERLAP,
        tophat_radius=TOPHAT_RADIUS,
        search_radius=SEARCH_RADIUS,
        n_workers=n_workers,
    )

    # Stage 3: Conservative Filtering
    logger.info("=" * 80)
    logger.info("Stage 3: Conservative Filtering")
    logger.info("=" * 80)
    n_before_filter = len(intensity_df)
    n_after_filter = n_before_filter
    if MIN_INTENSITY_THRESHOLD > 0:
        intensity_cols = [c for c in intensity_df.columns if c.startswith('cyc_')]
        intensity_max = intensity_df[intensity_cols].max(axis=1)
        n_before_filter = len(intensity_df)
        intensity_df = intensity_df[intensity_max >= MIN_INTENSITY_THRESHOLD].copy()
        n_after_filter = len(intensity_df)
        logger.info(f"Filtered: {n_before_filter} -> {n_after_filter} spots (threshold={MIN_INTENSITY_THRESHOLD})")
    else:
        logger.info("Filtering disabled (threshold=0)")

    # Stage 4: Deduplication
    logger.info("=" * 80)
    logger.info("Stage 4: Deduplication (block overlap)")
    logger.info("=" * 80)
    n_before_dedup = len(intensity_df)
    intensity_cols = [c for c in intensity_df.columns if c.startswith('cyc_')]
    if intensity_cols:
        intensity_df['_dedup_score'] = intensity_df[intensity_cols].sum(axis=1)
        intensity_df = deduplicate_dataframe(
            intensity_df,
            coordinate_columns=['Y', 'X'],
            threshold=DEDUPLICATE_THRESHOLD,
            sort_by='_dedup_score',
        )
        intensity_df = intensity_df.drop(columns=['_dedup_score'])
    else:
        intensity_df = deduplicate_dataframe(
            intensity_df,
            coordinate_columns=['Y', 'X'],
            threshold=DEDUPLICATE_THRESHOLD,
        )
    n_after_dedup = len(intensity_df)
    logger.info(f"Deduplication: {n_before_dedup} -> {n_after_dedup} spots "
                f"(removed {n_before_dedup - n_after_dedup})")

    # Stage 5: Save Output
    logger.info("=" * 80)
    logger.info("Stage 5: Saving Output")
    logger.info("=" * 80)

    intensity_df = intensity_df.reset_index(drop=True)

    # position.csv: index, Y, X
    position_df = intensity_df[['Y', 'X']].copy()
    position_df.insert(0, 'index', range(len(position_df)))
    position_file = read_dir / 'position.csv'
    position_df.to_csv(position_file, index=False)
    logger.info(f"Saved position: {position_file} ({len(position_df)} spots)")

    # intensity.csv: index, cyc_1_cy3, cyc_1_cy5, ...
    int_cols = [c for c in intensity_df.columns if c.startswith('cyc_')]
    intensity_output_df = intensity_df[int_cols].copy()
    intensity_output_df.insert(0, 'index', range(len(intensity_output_df)))
    intensity_file = read_dir / 'intensity.csv'
    intensity_output_df.to_csv(intensity_file, index=False)
    logger.info(f"Saved intensity: {intensity_file} ({len(intensity_output_df)} spots)")

    # Stage 6: QC
    logger.info("=" * 80)
    logger.info("Stage 6: QC Report")
    logger.info("=" * 80)
    try:
        from sprintseq.qc import generate_readout_qc
        generate_readout_qc(
            intensity_df=intensity_df,
            position_df=position_df,
            output_dir=read_dir,
            run_id=run_id,
            channels=channels,
            seq_cycle=seq_cycle,
            detection_stats=detection_stats,
            filter_stats={"n_before": n_before_filter, "n_after": n_after_filter, "threshold": MIN_INTENSITY_THRESHOLD},
            dedup_stats={"n_before": n_before_dedup, "n_after": n_after_dedup},
        )
        logger.info(f"  readout_qc.json + readout_qc.png saved to {read_dir}")
    except Exception as e:
        logger.warning(f"QC generation failed (non-fatal): {e}")

    logger.info("=" * 80)
    logger.info("Readout pipeline completed successfully!")
    logger.info(f"  position.csv: {position_file}")
    logger.info(f"  intensity.csv: {intensity_file}")
    logger.info("Next step: Run gene_calling to classify genes.")
    logger.info("=" * 80)

    return position_df, intensity_output_df


def main():
    """`python -m sprintseq.cli.readout` fallback entry point. The canonical CLI is `sprintseq readout` (see sprintseq.cli.main)."""
    parser = argparse.ArgumentParser(description='Stitched-image readout pipeline for SPRINTseq')
    parser.add_argument('--run-id', type=str, default=DEFAULT_RUN_ID,
                        help=f'Run ID to process (default: {DEFAULT_RUN_ID})')
    parser.add_argument('--n-workers', type=int, default=None,
                        help=f'Number of parallel workers (default: {N_WORKERS})')
    from sprintseq.cli import parse_cycles, parse_channels
    parser.add_argument('--detection-cycles', type=parse_cycles, default=None,
                        help=f'Cycles to detect spots in (e.g. "1,2,3,4" or "1-4,11" or "11"). '
                             f'Default: {DETECTION_CYCLES}')
    parser.add_argument('--seq-cycles', type=int, default=None,
                        help=f'Number of sequencing cycles (1..N) for intensity readout. '
                             f'Default: {SEQ_CYCLE}')
    parser.add_argument('--channels', type=parse_channels, default=None,
                        help=f'Comma-separated SBS channels. Default: {",".join(CHANNELS)}')
    parser.add_argument('--detection-method', type=str, default=None,
                        choices=list(DETECTION_METHODS),
                        help=f'Spot detection method. Default: {DETECTION_METHOD}')
    args = parser.parse_args()

    # Log to file
    read_dir = Path(BASE_DEST_DIRECTORY) / f'{args.run_id}_processed' / 'readout'
    read_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(read_dir / 'readout.log', encoding='utf-8')
    fh.setFormatter(logging.Formatter(_LOG_FMT))
    logging.getLogger().addHandler(fh)

    run_pipeline(args.run_id, n_workers=args.n_workers,
                 detection_cycles=args.detection_cycles,
                 seq_cycle=args.seq_cycles,
                 channels=args.channels,
                 detection_method=args.detection_method)


if __name__ == "__main__":
    main()
