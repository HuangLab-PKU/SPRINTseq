import os
from tqdm import trange
import numpy as np
import pandas as pd

import cv2
from skimage.io import imread
from skimage.feature import peak_local_max

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


def preprocess_image(image, sigma=1, tophat_radius=3):
    """Apply gaussian blur then optional disk white-tophat using cv2.

    Parameters:
        image: input image (numpy array)
        sigma: standard deviation of Gaussian blur (default 1)
        tophat_radius: radius of top hat transformation (default 3)
    
    Returns:
        preprocessed image, type same as input, suitable for peak_local_max and intensity sampling
    """
    if image.dtype != np.float32:
        image_f = image.astype(np.float32, copy=True)
    else:
        image_f = image.copy()
    
    # use cv2 for Gaussian blur
    # ksize=(0,0) means calculate kernel size automatically based on sigma
    blurred = cv2.GaussianBlur(image_f, ksize=(0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
    
    # use cv2 for top hat transformation (white hat transformation)
    if tophat_radius and tophat_radius > 0:
        # create elliptical structure element (similar to skimage's disk)
        kernel_size = 2 * int(tophat_radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        processed = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
    else:
        processed = blurred
    
    # convert back to original data type
    # if original image is integer type, round and clip
    if np.issubdtype(image.dtype, np.integer):
        processed = np.rint(processed).astype(np.int64)
        # clip to the range of original data type
        if image.dtype == np.uint8:
            np.clip(processed, 0, 255, out=processed)
            return processed.astype(np.uint8)
        elif image.dtype == np.uint16:
            np.clip(processed, 0, 65535, out=processed)
            return processed.astype(np.uint16)
        else:
            return processed.astype(image.dtype)
    else:
        return processed.astype(image.dtype)

def extract_coordinates(image, snr=4, quantile=0.96):
    meta = {}
    # Preprocess: Gaussian blur (sigma=1) then disk white tophat (radius=3)
    image = preprocess_image(image)
    # Find local maxima on preprocessed image
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

def get_coordinates(in_directory,channels=CHANNELS,cycle_num=CYCLE_NUM):
    # Collect coordinates from each image, then combine & unique once at the end.
    collected = []
    for i in trange(1, 1 + cycle_num):
        for channel in channels:
            im = imread(os.path.join(in_directory, f'cyc_{i}_{channel}.tif'))
            temp, _ = extract_coordinates(im, snr=SNRS[channel], quantile=QUANTILE)
            if temp.size: collected.append(temp)

    if collected:
        all_coords = np.vstack(collected)
        coordinates = np.unique(all_coords, axis=0)
    else:
        coordinates = np.empty((0, 2), dtype=int)

    return coordinates

def get_intensity_df(stc_directory, coordinates, cyc_num=SEQ_CYCLE):
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

    for cyc in range(1, cyc_num + 1):
        for channel in CHANNELS:
            im_path = os.path.join(stc_directory, f'cyc_{cyc}_{channel}.tif')
            image = imread(im_path)
            # Preprocessing is controlled by preprocess_image
            image = preprocess_image(image)
            intensity_df[f'cyc_{cyc}_{channel}'] = image[coords[:, 0], coords[:, 1]]

    return intensity_df

def get_seq_df(intensity_df, thresholds, cyc_num=SEQ_CYCLE):
    """
    Convert an intensity dataframe into a sequence dataframe by thresholding.

    intensity_df must contain columns 'Y','X' and 'cyc_{n}_{channel}' for
    each cycle and channel in CHANNELS. thresholds is a dict mapping channel->value.
    """
    coordinates = intensity_df[['Y', 'X']].to_numpy()

    # Build boolean calls per cycle/channel
    bool_df = pd.DataFrame({'Y': coordinates[:, 0], 'X': coordinates[:, 1]})
    for cyc in trange(1, cyc_num + 1, desc='Thresholding'):
        for channel in CHANNELS:
            col = f'cyc_{cyc}_{channel}'
            bool_df[col] = intensity_df[col] >= thresholds[channel]

    # Map boolean pairs to bases
    base_bool_map = {(True, True): 'A', (True, False): 'T', (False, True): 'C', (False, False): 'G'}
    base_df = pd.DataFrame({'Y': coordinates[:, 0], 'X': coordinates[:, 1]})
    for cyc in trange(1, cyc_num + 1, desc='Base calling'):
        c3 = bool_df[f'cyc_{cyc}_cy3']
        c5 = bool_df[f'cyc_{cyc}_cy5']
        pairs = list(zip(c3, c5))
        base_df[f'cyc_{cyc}'] = pd.Series(pairs).map(base_bool_map).values

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

    coordinates = get_coordinates(in_directory)
    print(f'Extracted {coordinates.shape[0]} puncta.')

    if coordinates.size == 0:
        # write empty files to keep downstream code happy
        empty_int_df = pd.DataFrame(columns=['Y', 'X'] + [f'cyc_{c}_{ch}' for c in range(1, SEQ_CYCLE + 1) for ch in CHANNELS])
        empty_int_df.to_csv(os.path.join(out_directory, 'intensity_filtered.csv'), index=False)
        empty_seq_df = pd.DataFrame(columns=['Y', 'X', 'Sequence'])
        empty_seq_df.to_csv(os.path.join(out_directory, RAW_SEQ_NAME), index=False)
        print('No coordinates found; wrote empty outputs.')
        return empty_seq_df

    intensity_df = get_intensity_df(in_directory, coordinates, cyc_num=SEQ_CYCLE)
    min_threshold = min(thresholds.values())
    # keep rows where any cycle/channel exceeds the minimum threshold
    intensity_df = intensity_df[intensity_df.iloc[:, 2:].max(axis=1) >= min_threshold]
    intensity_df.to_csv(os.path.join(out_directory, 'intensity_filtered.csv'), index=False)
    print(f'Obtained intensity sequence of {len(intensity_df)} puncta.')

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