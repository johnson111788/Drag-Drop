import os
import glob
import argparse
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

from utils.drag_drop import get_weak_label, get_marker, dilate_lesion
from utils.utils import load_img_label, save_output, cal_img_gradient


def watershed_core(input, label, N, M):
    
    pseudo_label = np.zeros_like(label, dtype=np.uint8)
    label_numeric, gt_N = ndimage.label(label)
    
    for segid in range(1, gt_N+1):
                
        label_binary = np.uint8(label_numeric == segid)
        marker = np.zeros_like(label_binary)
        
        # ******************** Sec. 2.1 Drag&Drop Initialization ********************
        center_x, center_y, sphere_radius = get_weak_label(label_binary)
        marker = get_marker(center_x, center_y, sphere_radius, N, marker)
        
        # ******************** Sec. 2.2 Drag&Drop Propagation ********************
        lesion = watershed(input, marker)-1
        
        # ******************** Sec. 2.3 Noise Reduction ******************** 
        dilated_lesion = dilate_lesion(lesion, sphere_radius, M)
        
        pseudo_label = pseudo_label | dilated_lesion
        
    pseudo_label[pseudo_label==3] = 1
    
    return pseudo_label
    

def watershed_engine(opt):
        
    imgpath_list =  glob.glob(opt.image_root + '*.jpg')
    
    for imgpath in imgpath_list:
    
        labelpath = imgpath.replace(opt.image_root, opt.label_root).replace('.jpg', '.png')
        tar_path = imgpath.replace(opt.image_root, opt.output_root).replace('.jpg', '.png')
        
        grayimage, label, image= load_img_label(imgpath, labelpath)
        gradient = cal_img_gradient(grayimage)
        
        pseudo_label = watershed_core(gradient, label, opt.N, opt.M)
        save_output(tar_path, pseudo_label, image)
            
                

if __name__ == '__main__':
    np.random.seed(1000)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_root', type=str, help='custom your image root',
        default='data/image/')
    parser.add_argument(
        '--label_root', type=str, help='custom your ground-truth root',
        default='data/label/')
    parser.add_argument(
        '--output_root', type=str, help='custom your prediction root',
        default='data/drag_drop/')
    
    # ******************** Hyper-parameters ********************
    parser.add_argument(
        '--N', type=float, help='the kernel size of dilation',
        default=0.2)
    parser.add_argument(
        '--M', type=float, help='the kernel size of dilation',
        default=0.5)
    opt = parser.parse_args()
    
    os.makedirs(opt.output_root, exist_ok=True)
    watershed_engine(opt)