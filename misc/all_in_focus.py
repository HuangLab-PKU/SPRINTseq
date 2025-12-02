import cv2
import numpy as np
from skimage import img_as_float64
from skimage import img_as_uint
from tqdm import tqdm

def update_focus(image,next_image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    next_grad_x = cv2.Sobel(next_image, cv2.CV_64F, 1, 0, ksize=5)
    next_grad_y = cv2.Sobel(next_image, cv2.CV_64F, 0, 1, ksize=5)
    grad = np.abs(grad_x) + np.abs(grad_y)
    next_grad = np.abs(next_grad_x) + np.abs(next_grad_y)
    matrix = grad - next_grad
    filter_matrix = np.ones((3, 3))
    image_filter = cv2.filter2D(matrix, -1, filter_matrix, borderType=cv2.BORDER_CONSTANT)
    # β parameter - the best results for β about 0.1
    beta = 0.1
    image_filter = 1/(1+np.exp(-beta*image_filter))
    result = image_filter * image + (1-image_filter) * next_image
    return result

def all_in_focus(img_list):
    for i,file_name in enumerate(img_list):
        if i==0:
            image = cv2.imread(file_name, -cv2.IMREAD_ANYDEPTH)
            image = img_as_float64(image)
            continue
        next_image = cv2.imread(file_name, -cv2.IMREAD_ANYDEPTH)
        next_image = img_as_float64(next_image)
        image = update_focus(image,next_image)
    result = img_as_uint(image)
    return result