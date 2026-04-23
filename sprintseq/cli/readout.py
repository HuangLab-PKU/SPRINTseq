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

def _get_stitched_image_path(stc_dir, cyc, channel):
    """Return stitched image path: stc_dir / cyc_{cyc}_{channel}.tif"""
    return Path(stc_dir) / f'cyc_{cyc}_{channel}.tif'


def detect_all_spots(
    stc_dir,
    channels=CHANNELS,
    detection_cycles=None,
    block_size=BLOCK_SIZE,
    block_overlap=BLOCK_OVERLAP,
    detection_method=DETECTION_METHOD,
    snrs=SNRS,
    n_workers=N_WORKERS,
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

    Returns
    -------
    np.ndarray
        (N, 2) array of unique global (Y, X) coordinates.
    """
    stc_dir = Path(stc_dir)
    by, bx = block_size
    if detection_cycles is None:
        detection_cycles = DETECTION_CYCLES

    # Count total blocks for progress bar
    n_total_blocks = 0
    for cyc in detection_cycles:
        for channel in channels:
            img_path = _get_stitched_image_path(stc_dir, cyc, channel)
            if not img_path.exists():
                logger.warning(f"Stitched image not found: {img_path}, skipping")
                continue
            with tifffile.TiffFile(str(img_path)) as t:
                sh = t.pages[0].shape
            h, w = (sh[0], sh[1]) if len(sh) == 2 else (sh[1], sh[2])
            n_total_blocks += len(block_starts(h, w, block_size, block_overlap))

    # Build detection kwargs per channel
    def _get_detection_kwargs(channel):
        if detection_method == 'spotiflow':
            return {'prob_thresh': None, 'device': None}  # let get_spot_coordinates auto-detect
        else:
            return {'snr': snrs.get(channel, 3.0), 'tophat_radius': TOPHAT_RADIUS}

    # Lazy iterator: open memmap per image, yield blocks as numpy arrays
    def _detection_task_iter():
        for cyc in detection_cycles:
            for channel in channels:
                img_path = _get_stitched_image_path(stc_dir, cyc, channel)
                if not img_path.exists():
                    continue
                detection_kwargs = _get_detection_kwargs(channel)
                img = tifffile.memmap(str(img_path))
                if img.ndim == 3:
                    img = img[0]
                h, w = img.shape
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
        return np.empty((0, 2), dtype=np.float64)

    # Log per-channel counts (before exact-duplicate removal)
    coords_by_channel = defaultdict(list)
    for channel, coords in results:
        coords_by_channel[channel].append(coords)
    for channel in channels:
        if channel in coords_by_channel:
            ch_coords = np.vstack(coords_by_channel[channel])
            logger.info(f"  {channel}: {len(ch_coords)} spots detected (raw, before dedup)")

    # Combine and remove exact duplicates
    all_coords = np.vstack([coords for _, coords in results])
    logger.info(f"Total coordinates before deduplication: {len(all_coords)}")
    coords_rounded = np.round(all_coords).astype(np.int32)
    _, unique_indices = np.unique(coords_rounded, axis=0, return_index=True)
    unique_coords = all_coords[unique_indices]
    logger.info(f"Total coordinates after removing exact duplicates: {len(unique_coords)}")

    return unique_coords


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
        if _get_stitched_image_path(stc_dir, cyc, ch).exists()
    )
    n_intensity_total = len(coords_in_block) * n_existing_images

    # Lazy iterator: open memmap per image, yield blocks with spots
    def _intensity_task_iter():
        for cyc in range(1, seq_cycle + 1):
            for channel in channels:
                col_name = f'cyc_{cyc}_{channel}'
                img_path = _get_stitched_image_path(stc_dir, cyc, channel)
                if not img_path.exists():
                    logger.warning(f"Stitched image not found: {img_path}, filling with NaN")
                    continue
                img = tifffile.memmap(str(img_path))
                if img.ndim == 3:
                    img = img[0]
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
                 channels=None):
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
    """
    # Copy list defaults so callers can't mutate the module-level state via
    # the returned reference.
    n_workers = N_WORKERS if n_workers is None else n_workers
    detection_cycles = list(DETECTION_CYCLES) if detection_cycles is None else list(detection_cycles)
    seq_cycle = SEQ_CYCLE if seq_cycle is None else seq_cycle
    channels = list(CHANNELS) if channels is None else list(channels)

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
    logger.info(f"Detection: {DETECTION_METHOD}, cycles {detection_cycles}, channels {channels}")
    logger.info(f"Intensity: tophat (radius={TOPHAT_RADIUS}, search={SEARCH_RADIUS}), "
                f"cycles 1-{seq_cycle}, channels {channels}")
    logger.info(f"Block size: {BLOCK_SIZE}, overlap: {BLOCK_OVERLAP}")
    logger.info(f"Workers: {n_workers}")
    logger.info("=" * 80)

    # Stage 1: Spot Detection
    logger.info("=" * 80)
    logger.info("Stage 1: Spot Detection")
    logger.info("=" * 80)
    unique_coords = detect_all_spots(
        stc_dir,
        channels=channels,
        detection_cycles=detection_cycles,
        block_size=BLOCK_SIZE,
        block_overlap=BLOCK_OVERLAP,
        detection_method=DETECTION_METHOD,
        snrs=SNRS,
        n_workers=n_workers,
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
    if MIN_INTENSITY_THRESHOLD > 0:
        intensity_cols = [c for c in intensity_df.columns if c.startswith('cyc_')]
        intensity_max = intensity_df[intensity_cols].max(axis=1)
        n_before = len(intensity_df)
        intensity_df = intensity_df[intensity_max >= MIN_INTENSITY_THRESHOLD].copy()
        n_after = len(intensity_df)
        logger.info(f"Filtered: {n_before} -> {n_after} spots (threshold={MIN_INTENSITY_THRESHOLD})")
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
                 channels=args.channels)


if __name__ == "__main__":
    main()
