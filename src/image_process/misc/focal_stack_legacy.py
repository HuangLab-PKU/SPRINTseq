import numpy as np
import cv2 
from skimage import img_as_float
from glob import glob

def get_focus(im, ksize):
    im_blur = cv2.blur(im,(ksize,ksize))
    im_subtract = (im-im_blur)**2
    return cv2.blur(im_subtract,(ksize,ksize))

def gauss_interpolation(im_seq):
    step = 2
    x,y,p = im_seq.shape
    x_ind,y_ind = np.indices((x,y))
    im_max = np.max(im_seq, axis=2)
    max_ind = np.argmax(im_seq,axis=2)
    x2 = max_ind.copy()
    x2[max_ind<=step] = step
    x2[max_ind>=step] = p-step-1
    x3 = x2 + step
    x1 = x2 - step
    y1 = np.log(im_seq[x_ind,y_ind,x1])
    y2 = np.log(im_seq[x_ind,y_ind,x2])
    y3 = np.log(im_seq[x_ind,y_ind,x3])
    c = ((y1-y2)*(x2-x3)-(y2-y3)*(x1-x2))/((x1**2-x2**2)*(x2-x3)-(x2**2-x3**2)*(x1-x2))
    b = ((y2-y3)-c*(x2-x3)*(x2+x3))/(x2-x3)
    u = np.abs(b/(2*c))
    a = y1 - b * x1 - c * x1**2
    return u,a,c,im_max

def read_im_seq(imlist):
    for i,im_name in enumerate(imlist):
        im = cv2.imread(im_name, -cv2.IMREAD_ANYDEPTH)
        if i == 0:
            x,y = im.shape
            im_seq = np.zeros((x,y,len(imlist)))
            f_measure = np.zeros((x,y,len(imlist)))
        im_seq[:,:,i] = im
        f_measure[:,:,i] = get_focus(img_as_float(im),9)
    return im_seq,f_measure
    
def focal_stack(imlist,nhsize=9,focus=1,alpha=0.2,sth=13):
    im_seq,f_measure = read_im_seq(imlist)
    x,y,p = im_seq.shape
    u,a,c,im_max = gauss_interpolation(f_measure)
    err = np.sum(np.abs(f_measure - np.exp(a[:,:,np.newaxis]+(u**2*np.abs(c))[:,:,np.newaxis]-((np.arange(len(imlist))+1)+u[:,:,np.newaxis])**2*np.abs(c)[:,:,np.newaxis])),axis=2)
    inv_psnr = err / (im_max * p)
    S = -40 * np.log10(inv_psnr)
    S[np.isnan(S)] = np.nanmin(S)
    phi = 0.5*(1+np.tanh(alpha*(S-sth)))/alpha
    phi = cv2.blur(phi,(3,3))
    phi = phi[:,:,np.newaxis] * np.ones(p)
    f_measure = f_measure / im_max[:,:,np.newaxis]
    f_measure = 0.5 + 0.5 * np.tanh(phi*(f_measure-1))
    nm_fac = np.sum(f_measure, axis=2)
    im_out = np.uint16(np.sum(im_seq*f_measure,axis=2)/nm_fac)
    return im_out

def main():
    imlist = glob('/Users/Leon/Downloads/fstack_legacy_test/cy3/*.tif')
    cv2.imwrite('sample_output.tif',focal_stack(imlist))

if __name__ == "__main__":
    main()