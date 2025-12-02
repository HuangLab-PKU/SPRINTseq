import os
import numpy as np
from skimage.io import imread
from skimage.io import imsave
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu
from skimage.morphology import remove_small_objects
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import measure

MIN_CELL_SIZE = 400

def segment_cell(im,return_image=True):
    thresholds = threshold_multiotsu(im, classes=3)
    cells = im > thresholds[0]
    cells = remove_small_objects(cells, MIN_CELL_SIZE)
    distance = ndi.distance_transform_edt(cells)
    coordinates = peak_local_max(distance, min_distance=7)
    if return_image:
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coordinates.T)] = True
        markers = measure.label(mask)
        segmented = watershed(-distance,markers,mask=cells)
        return coordinates,segmented
    else:
        return coordinates

def main():
    im = imread('./FocalStack_072.tif')
    coordinates,segmented = segment_cell(im)
    np.savetxt('coordinates_0920.txt',coordinates,fmt='%d')
    imsave('segmented_0920.tif',segmented,check_contrast=False)

if __name__ == "__main__":
    main()
