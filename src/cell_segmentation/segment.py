from pathlib import Path
from tqdm import tqdm
from math import ceil
import numpy as np
import cv2
import scipy.ndimage as ndi
from scipy.spatial import KDTree
from skimage.io import imread
from skimage.io import imsave
from skimage import measure
from skimage.filters import threshold_local
from skimage.morphology import remove_small_objects
from skimage.morphology import disk
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


MIN_REGION = 400
MIN_CELL_SIZE = 400
BLOCK_SIZE = 20000
BLOCK_STRIDE = BLOCK_SIZE - 100
PROCESSED_BASE_DIR = Path('/mnt/data/local_processed_data')


def segment_cell(im,offset_value=255,BLOCK_SIZE=251):
    threshold_im = threshold_local(im,BLOCK_SIZE,offset=-offset_value)
    threshold_sup = np.quantile(threshold_im,0.98)
    threshold_im[threshold_im>threshold_sup] = threshold_sup
    bool_mask = im > threshold_im
    bool_mask = remove_small_objects(bool_mask, MIN_REGION)
    cells = np.zeros(im.shape,dtype=np.uint8)
    cells[im>threshold_im] = 1000
    cells = cv2.dilate(cells,disk(3))
    cells = cv2.erode(cells,disk(2))
    cells = cells > 0
    cells = remove_small_objects(cells, MIN_REGION)
    distance = ndi.distance_transform_edt(cells)
    coordinates = peak_local_max(distance, min_distance=7)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coordinates.T)] = True
    markers = measure.label(mask)
    segmented = watershed(-distance,markers,mask=cells)
    unique, counts = np.unique(segmented, return_counts=True)
    small_labels = unique[counts<MIN_CELL_SIZE]
    segmented[np.isin(segmented,small_labels)] = 0.0
    coordinates = coordinates[segmented[coordinates[:,0],coordinates[:,1]]!=0.0]
    return coordinates,segmented


def block_segment(im,out_dir,output=True,output_original=True):
    overlap = BLOCK_STRIDE - BLOCK_SIZE
    y,x = im.shape
    y_steps = ceil((y-overlap) / BLOCK_STRIDE)
    x_steps = ceil((x-overlap) / BLOCK_STRIDE)
    print(f'Segmenting image with {im.shape}...')
    print(f'A total of {y_steps} x {x_steps} blocks...')
    for y_step in tqdm(range(y_steps)):
        for x_step in range(x_steps):
            block = im[y_step*BLOCK_STRIDE:y_step*BLOCK_STRIDE+BLOCK_SIZE,x_step*BLOCK_STRIDE:x_step*BLOCK_STRIDE+BLOCK_SIZE]
            coordinates,segmented = segment_cell(block)
            coordinates += [y_step*BLOCK_STRIDE,x_step*BLOCK_STRIDE]
            np.savetxt(Path(out_dir)/f'centroids_y_{y_step}_x_{x_step}.csv',coordinates,fmt='%d',delimiter=',')
            if output:
                imsave(Path(out_dir)/f'segmented_y_{y_step}_x_{x_step}.tif',segmented,check_contrast=False)
            if output_original:
                imsave(Path(out_dir)/f'centroids_y_{y_step}_x_{x_step}_original.tif',block,check_contrast=False)


def remove_duplicates(coordinates,distance=5):
    tree = KDTree(coordinates)
    pairs = tree.query_pairs(distance)
    neighbors = {}
    for i,j in pairs:
        if i not in neighbors:
            neighbors[i] = set([j])
        else:
            neighbors[i].add(j)
        if j not in neighbors:
            neighbors[j] = set([i])
        else:
            neighbors[j].add(i)
    keep = []
    discard = set()
    nodes = set([s[0] for s in pairs]+[s[1] for s in pairs])
    for node in nodes:
        if node not in discard:
            keep.append(node)
            discard.update(neighbors.get(node,set()))
    coordinates_simplified = np.delete(coordinates, list(discard), axis=0)
    return coordinates_simplified

def combine_centroids(in_dir):
    centroids_all = None
    centroids_list = list(Path(in_dir).glob('centroids_y_*_x_*.csv'))
    for path in centroids_list:
        centroids = np.loadtxt(path, delimiter=',',dtype=int)
        if len(centroids) == 0:
            continue
        elif len(centroids.shape) == 1:
            centroids = centroids[np.newaxis,:]
        if centroids_all is None:
            centroids_all = centroids
        else:
            centroids_all = np.unique(np.concatenate((centroids_all,centroids),axis=0),axis=0)
    print(f'Total number of centroids: {centroids_all.shape[0]}')
    centroids_all = remove_duplicates(centroids_all)
    print(f'Number of unique centroids: {centroids_all.shape[0]}')
    np.savetxt(Path(in_dir)/'centroids_all.csv',centroids_all,fmt='%d',delimiter=',')


def segment_pipeline(run_id):
    base_dir = PROCESSED_BASE_DIR / f'{run_id}_processed'
    stc_dir = base_dir / 'stitched'
    seg_dir = base_dir / 'segmented'
    seg_dir.mkdir(exist_ok=True)
    cell_im_name = 'cyc_10_DAPI.tif'
    im = imread(stc_dir/cell_im_name)
    block_segment(im,seg_dir)
    combine_centroids(seg_dir)


#if __name__ == '__main__':
#    pass

segment_pipeline('20221207_OSCC_3_thermo')