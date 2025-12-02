import os

from tqdm import trange
import numpy as np
import pandas as pd
# Compatibility shim: newer NumPy versions removed deprecated aliases like np.bool,
# which older versions of scikit-image still reference. Provide safe aliases
# so scikit-image code that checks `np.bool`, `np.int`, etc. won't raise.
for _alias, _type in (('bool', bool), ('int', int), ('float', float), ('object', object), ('str', str)):
    # Use np.__dict__ to check for the attribute to avoid triggering
    # NumPy's deprecation FutureWarnings that occur when accessing
    # attributes like np.bool via getattr/hasattr.
    if _alias not in np.__dict__:
        setattr(np, _alias, _type)

from skimage.io import imread
from skimage.feature import peak_local_max
import math
import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import tifffile
import cv2
import json

from ..src.gene_calling.reference_check import check_sequence
from ..src.gene_calling.mapping import map_barcode
from ..src.gene_calling.mapping import unstack_plex

# basic
BASE_DIRECTORY = 'path_to_raw_images_archive'
BASE_DEST_DIRECTORY = 'path_to_processed_images'
RUN_ID = 'example'
CHANNELS = ['cy3', 'cy5']

# spot detection
CYCLE_NUM = 4
QUANTILE = 0.1
SNRS = {'cy3': 1.0, 'cy5': 1.0}

# sequence readout
SEQ_CYCLE = 10
THRESHOLDS = {'cy3': 50, 'cy5': 25}
RAW_SEQ_NAME = 'raw_sequence.csv'
CHECKED_NAME = 'ref_checked.csv'
REF_FOLDER = 'path_to_reference_file'
REF_FILE = os.path.join(REF_FOLDER, 'example.csv')

# ---------------------- Embedded tiled cv2 preprocessor ----------------------
logger = logging.getLogger('readout_preproc')


def _process_tile_task(args):
    (in_path, out_path, read_y0, read_y1, read_x0, read_x1,
     write_y0, write_y1, write_x0, write_x1, sigma, tophat_radius) = args

    in_mm = tifffile.memmap(in_path, mode='r')
    out_mm = tifffile.memmap(out_path, mode='r+')

    tile = in_mm[read_y0:read_y1, read_x0:read_x1]
    tile_f = tile.astype(np.float32, copy=False)

    blurred = cv2.GaussianBlur(tile_f, ksize=(0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)

    if tophat_radius and tophat_radius > 0:
        ksz = 2 * int(tophat_radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        processed = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
    else:
        processed = blurred

    inner_y0 = write_y0 - read_y0
    inner_y1 = inner_y0 + (write_y1 - write_y0)
    inner_x0 = write_x0 - read_x0
    inner_x1 = inner_x0 + (write_x1 - write_x0)

    out_tile = np.rint(processed[inner_y0:inner_y1, inner_x0:inner_x1]).astype(np.int64)
    np.clip(out_tile, 0, 65535, out=out_tile)

    out_mm[write_y0:write_y1, write_x0:write_x1] = out_tile.astype(np.uint16)

    del in_mm
    del out_mm

    return (write_y0, write_y1, write_x0, write_x1)


def preprocess_image(in_path, out_path, sigma=1.0, tophat_radius=3,
                       tile_height=2000, tile_width=None, workers=16):
    t0 = time.time()
    with tifffile.TiffFile(in_path) as tif:
        page = tif.pages[0]
        H, W = page.shape

    if tile_width is None:
        tile_width = W

    overlap = int(math.ceil(max(3 * sigma, tophat_radius)))
    logger.info('Image shape: %d x %d, tile: %d x %d, overlap: %d, workers: %d', H, W, tile_height, tile_width, overlap, workers)

    if os.path.exists(out_path):
        os.remove(out_path)
    # Create a proper TIFF file with correct headers by writing an all-zero array.
    # This ensures worker processes can open the file with tifffile.memmap in 'r+' mode
    # without encountering 'not a TIFF file' errors on some platforms/filesystems.
    zeros = np.zeros((H, W), dtype=np.uint16)
    tifffile.imwrite(out_path, zeros, bigtiff=True)
    # free memory
    del zeros

    tasks = []
    for y0 in range(0, H, tile_height):
        y1 = min(y0 + tile_height, H)
        read_y0 = max(0, y0 - overlap)
        read_y1 = min(H, y1 + overlap)
        for x0 in range(0, W, tile_width):
            x1 = min(x0 + tile_width, W)
            read_x0 = max(0, x0 - overlap)
            read_x1 = min(W, x1 + overlap)
            task = (in_path, out_path, read_y0, read_y1, read_x0, read_x1,
                    y0, y1, x0, x1, sigma, tophat_radius)
            tasks.append(task)

    total_tasks = len(tasks)
    logger.info('Total tiles to process: %d', total_tasks)

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = [exe.submit(_process_tile_task, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            if done % 10 == 0 or done == total_tasks:
                logger.info('Finished %d / %d tiles', done, total_tasks)

    logger.info('Done. Elapsed %.1f s', time.time() - t0)

def ensure_preprocessed_images(in_directory, sigma=1.0, tophat_radius=3, tile=2000, workers=None, overwrite=False, cycles=None):
    """Create preprocessed images under `stitched/preproc/` for each cycle/channel.

    This function will create files named `cyc_{i}_{channel}_preproc.tif` in
    a `preproc` subdirectory of `in_directory`.
    """
    if workers is None:
        try:
            workers = max(1, os.cpu_count() - 1)
        except Exception:
            workers = 4

    preproc_dir = os.path.join(in_directory, 'preproc')
    os.makedirs(preproc_dir, exist_ok=True)

    # Determine which cycle indices to prepare. By default, prepare up to the
    # maximum of spot-detection cycles and sequence cycles so downstream
    # intensity-reading (which may use SEQ_CYCLE) has images available.
    if cycles is None:
        try:
            cycles = max(int(CYCLE_NUM), int(SEQ_CYCLE))
        except Exception:
            cycles = CYCLE_NUM

    created = []
    for cyc in range(1, cycles + 1):
        for channel in CHANNELS:
            inp = os.path.join(in_directory, f'cyc_{cyc}_{channel}.tif')
            out = os.path.join(preproc_dir, f'cyc_{cyc}_{channel}_preproc.tif')
            if not os.path.exists(inp):
                continue
            if os.path.exists(out) and not overwrite:
                created.append(out)
                continue

            # process
            preprocess_image(inp, out, sigma=sigma, tophat_radius=tophat_radius, tile_height=tile, tile_width=None, workers=workers)
            created.append(out)

    return created


def _select_image_path(stc_directory, cyc, channel):
    pre = os.path.join(stc_directory, 'preproc', f'cyc_{cyc}_{channel}_preproc.tif')
    orig = os.path.join(stc_directory, f'cyc_{cyc}_{channel}.tif')
    return pre if os.path.exists(pre) else orig


def _preproc_meta_path(tif_path):
    return tif_path + '.meta.json'


def _read_preproc_mean(tif_path):
    meta = _preproc_meta_path(tif_path)
    if os.path.exists(meta):
        try:
            with open(meta, 'r') as fh:
                d = json.load(fh)
            return float(d.get('mean'))
        except Exception:
            return None
    return None

# ---------------------- end preprocessor -------------------------------------


# ---------------------- Tile-based parallel detection ------------------------
def _detect_tile_task(args):
    (in_path, read_y0, read_y1, read_x0, read_x1,
     min_distance, threshold_abs) = args

    mm = tifffile.memmap(in_path, mode='r')
    tile = mm[read_y0:read_y1, read_x0:read_x1]
    tile_f = tile.astype(np.float32, copy=False)

    # Use skimage's local max on the tile (returns coordinates relative to tile)
    if threshold_abs is None:
        peaks = peak_local_max(tile_f, min_distance=int(min_distance))
    else:
        peaks = peak_local_max(tile_f, min_distance=int(min_distance), threshold_abs=float(threshold_abs))

    if peaks.size == 0:
        del mm
        return np.empty((0, 2), dtype=int)

    ys = peaks[:, 0]
    xs = peaks[:, 1]
    if ys.size == 0:
        del mm
        return np.empty((0, 2), dtype=int)

    ys_global = ys + read_y0
    xs_global = xs + read_x0
    coords = np.column_stack((ys_global, xs_global))

    del mm
    return coords


def detect_candidates_for_image(in_path, min_distance=2, tile_height=2000, tile_width=None, workers=None, threshold_abs=None):
    """Detect local-max candidates across a (preprocessed) image using tiled parallelism.

    Returns an (N,2) int array of (Y,X) coordinates (unique, deduplicated).
    """
    if workers is None:
        try: 
            workers = max(1, os.cpu_count() - 1)
        except Exception:
            workers = 4

    with tifffile.TiffFile(in_path) as tif:
        page = tif.pages[0]
        H, W = page.shape

    if tile_width is None:
        tile_width = W

    overlap = int(math.ceil(2 * min_distance))

    tasks = []
    for y0 in range(0, H, tile_height):
        y1 = min(y0 + tile_height, H)
        read_y0 = max(0, y0 - overlap)
        read_y1 = min(H, y1 + overlap)
        for x0 in range(0, W, tile_width):
            x1 = min(x0 + tile_width, W)
            read_x0 = max(0, x0 - overlap)
            read_x1 = min(W, x1 + overlap)
            task = (in_path, read_y0, read_y1, read_x0, read_x1, min_distance, threshold_abs)
            tasks.append(task)

    coords_list = []
    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = [exe.submit(_detect_tile_task, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            if res.size:
                coords_list.append(res)

    if not coords_list:
        return np.empty((0, 2), dtype=int)

    all_coords = np.vstack(coords_list)
    # deduplicate integer coordinates
    uniq = np.unique(all_coords, axis=0)
    return uniq

# ---------------------- end detection ---------------------------------------

def extract_coordinates(image, snr=4, quantile=0.96):
    meta = {}
    # image is expected to be preprocessed (process_large_tiff) before calling
    # this function. Find local maxima on the provided preprocessed image.
    coordinates = peak_local_max(image, min_distance=2, threshold_abs=snr * np.mean(image))
    meta['Coordinates brighter than given SNR'] = coordinates.shape[0]
    meta['Image mean intensity'] = float(np.mean(image))
    intensities = image[coordinates[:,0], coordinates[:,1]]
    meta[f'{quantile} quantile'] = float(np.quantile(intensities, quantile))
    threshold = np.quantile(intensities, quantile)
    # Filter coordinates by the intensity in the preprocessed image
    coordinates = coordinates[image[coordinates[:,0],coordinates[:,1]] > threshold]
    meta['Final spots count'] = coordinates.shape[0]
    return coordinates, meta

def get_coordinates(in_directory, channels=CHANNELS, cycle_num=CYCLE_NUM):
    # Collect coordinates from each image, then combine & unique once at the end.
    collected = []
    # Ensure preprocessed images exist (automatic mode) and placed under stitched/preproc/
    try:
        ensure_preprocessed_images(in_directory, sigma=1.0, tophat_radius=3, tile=2000, cycles=cycle_num)
    except Exception as e:
        # Fail fast per your request; re-raise
        raise

    for cyc in trange(1, 1 + cycle_num, desc='Spot detection'):
        for channel in channels:
            im_path = _select_image_path(in_directory, cyc, channel)
            if not os.path.exists(im_path):
                continue
            # Use memmap to sample intensities and know mean
            mm = tifffile.memmap(im_path, mode='r')
            img_mean = float(np.mean(mm))
            threshold_abs = SNRS[channel] * img_mean

            # Detect candidate peaks using tiled parallel detection (uses threshold_abs to reduce noise)
            coords = detect_candidates_for_image(im_path, min_distance=2, tile_height=2000, tile_width=None, workers=None, threshold_abs=threshold_abs)
            if coords.size == 0:
                continue

            # Compute per-image quantile on the preprocessed intensities and filter
            intensities = mm[coords[:, 0], coords[:, 1]]
            q = float(np.quantile(intensities, QUANTILE))
            coords = coords[intensities > q]
            if coords.size:
                collected.append(coords)

    if collected:
        all_coords = np.vstack(collected)
        coordinates = np.unique(all_coords, axis=0)
    else:
        coordinates = np.empty((0, 2), dtype=int)
    
    return coordinates

def get_intensity_df(in_directory, coordinates, cyc_num=SEQ_CYCLE):
    """
    Build an intensity dataframe for provided coordinates.

    Parameters
    - stc_directory: path to stitched images
    - coordinates: Nx2 array-like of (Y,X) coordinates
    - cyc_num: number of cycles to read

    Returns a pandas.DataFrame with columns ['Y','X', 'cyc_1_cy3', ...]
    """
    coords = np.asarray(coordinates)
    # If no coordinates, return an empty dataframe with the expected columns
    cols = ['Y', 'X'] + [f'cyc_{c}_{ch}' for c in range(1, cyc_num + 1) for ch in CHANNELS]
    if coords.size == 0:
        return pd.DataFrame(columns=cols)

    intensity_df = pd.DataFrame({'Y': coords[:, 0], 'X': coords[:, 1]})
    
    # Ensure preprocessed images exist (automatic mode) and placed under stitched/preproc/
    try:
        ensure_preprocessed_images(in_directory, sigma=1.0, tophat_radius=3, tile=2000, cycles=cyc_num)
    except Exception as e:
        # Fail fast per your request; re-raise
        raise
    
    for cyc in range(1, cyc_num + 1):
        for channel in CHANNELS:
            im_path = _select_image_path(in_directory, cyc, channel)
            image = imread(im_path)
            intensity_df[f'cyc_{cyc}_{channel}'] = image[coords[:, 0], coords[:, 1]]

    return intensity_df

def get_seq_df(intensity_df, thresholds, cyc_num=SEQ_CYCLE):
    """
    Convert an intensity dataframe into a sequence dataframe by thresholding.

    intensity_df must contain columns 'Y','X' and 'cyc_{n}_{channel}' for
    each cycle and channel in CHANNELS. thresholds is a dict mapping channel->value.
    """
    coordinates = intensity_df[['Y', 'X']].to_numpy()

    # Build boolean calls per cycle/channel. Be robust to missing/NaN values
    bool_df = pd.DataFrame({'Y': coordinates[:, 0], 'X': coordinates[:, 1]})
    for cyc in trange(1, cyc_num + 1, desc='Thresholding'):
        for channel in CHANNELS:
            col = f'cyc_{cyc}_{channel}'
            # If intensity column is missing or contains NaN, treat as 0 (below threshold)
            if col not in intensity_df.columns:
                bool_series = pd.Series(False, index=range(len(coordinates)))
            else:
                bool_series = (intensity_df[col].fillna(0) >= thresholds[channel])
            bool_df[col] = bool_series.astype(bool).values

    # Map boolean pairs to bases
    base_bool_map = {(True, True): 'A', (True, False): 'T', (False, True): 'C', (False, False): 'G'}
    base_df = pd.DataFrame({'Y': coordinates[:, 0], 'X': coordinates[:, 1]})
    for cyc in trange(1, cyc_num + 1, desc='Base calling'):
        c3 = bool_df[f'cyc_{cyc}_cy3']
        c5 = bool_df[f'cyc_{cyc}_cy5']
        # Ensure pairs are Python bool tuples so mapping won't produce NaN
        pairs = [(bool(a), bool(b)) for a, b in zip(c3, c5)]
        base_df[f'cyc_{cyc}'] = pd.Series(pairs).map(base_bool_map).astype(str).values

    seq_df = pd.DataFrame({'Y': coordinates[:, 0], 'X': coordinates[:, 1]})
    seq_df['Sequence'] = base_df[[f'cyc_{i}' for i in range(1, cyc_num + 1)]].agg(''.join, axis=1)
    return seq_df

def extract_raw_sequence(in_directory, out_directory, thresholds):
    """
    Extract intensity sequences and call bases for detected coordinates.

    Parameters
    - in_directory: path to stitched images (passed to get_coordinates/get_intensity_df)
    - out_directory: directory to write intermediate and output CSVs
    - thresholds: dict mapping channel->threshold for base calling

    Returns: seq_df (pandas.DataFrame) with columns ['Y','X','Sequence']
    """
    os.makedirs(out_directory, exist_ok=True)

    # print('Extracting coordinates...')
    # coordinates = get_coordinates(in_directory)
    # print(f'Extracted {coordinates.shape[0]} puncta.')

    # if coordinates.size == 0:
    #     # write empty files to keep downstream code happy
    #     empty_int_df = pd.DataFrame(columns=['Y', 'X'] + [f'cyc_{c}_{ch}' for c in range(1, SEQ_CYCLE + 1) for ch in CHANNELS])
    #     empty_int_df.to_csv(os.path.join(out_directory, 'intensity_filtered.csv'), index=False)
    #     empty_seq_df = pd.DataFrame(columns=['Y', 'X', 'Sequence'])
    #     empty_seq_df.to_csv(os.path.join(out_directory, RAW_SEQ_NAME), index=False)
    #     print('No coordinates found; wrote empty outputs.')
    #     return empty_seq_df

    # print('Building intensity dataframe...')
    # intensity_df = get_intensity_df(in_directory, coordinates, cyc_num=SEQ_CYCLE)
    # min_threshold = min(thresholds.values())
    # # keep rows where any cycle/channel exceeds the minimum threshold
    # intensity_df = intensity_df[intensity_df.iloc[:, 2:].max(axis=1) >= min_threshold]
    # intensity_df.to_csv(os.path.join(out_directory, 'intensity_filtered.csv'), index=False)
    # print(f'Obtained intensity sequence of {len(intensity_df)} puncta.')
    intensity_df = pd.read_csv(os.path.join(out_directory, 'intensity_filtered.csv'))

    # use the in-memory filtered dataframe rather than re-reading from disk
    seq_df = get_seq_df(intensity_df, thresholds, cyc_num=SEQ_CYCLE)
    print('Obtained raw sequence dataframe.')

    seq_df.to_csv(os.path.join(out_directory, RAW_SEQ_NAME), index=False)
    return seq_df


def main(run_id):
    dest_directory = os.path.join(BASE_DEST_DIRECTORY, f'{run_id}_processed')
    stc_directory = os.path.join(dest_directory, 'stitched')
    read_directory = os.path.join(dest_directory, 'readout')
    try: os.mkdir(read_directory)
    except FileExistsError: pass
    
    extract_raw_sequence(stc_directory, read_directory, THRESHOLDS)

    checked_df = check_sequence(os.path.join(read_directory, RAW_SEQ_NAME), REF_FILE)
    checked_df.to_csv(os.path.join(read_directory, CHECKED_NAME), index=False)
    
    map_df = map_barcode(os.path.join(read_directory, CHECKED_NAME), REF_FILE)
    map_df = unstack_plex(map_df)
    map_df.to_csv(os.path.join(read_directory, 'mapped_genes.csv'), index=False)


if __name__ == "__main__":
    main(RUN_ID)