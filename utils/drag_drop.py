import numpy as np
from scipy import ndimage

from utils.utils import sample_spherical


def get_weak_label(label_binary):
    
    lesion_index = np.where(label_binary==1)
    center_x, center_y = (np.max(lesion_index[0])+np.min(lesion_index[0]))//2, (np.max(lesion_index[1])+np.min(lesion_index[1]))//2
    
    width, height = np.max(lesion_index[0])-np.min(lesion_index[0]), np.max(lesion_index[1])-np.min(lesion_index[1])
    
    sphere_radius = int(max(width, height)//2)
    
    return center_x, center_y, sphere_radius
    

def get_marker(center_x, center_y, sphere_radius, N, marker):
            
    pos_marker_radius = sphere_radius*N
    
    # ******************** Positive Marker ********************
    noisex, noisey = np.int32(np.random.normal(0, 1, 1)), np.int32(np.random.normal(0, 1, 1))
    marker[center_x+noisex[0]-round(pos_marker_radius): center_x+noisex[0]+round(pos_marker_radius)+1,
        center_y+noisey[0]-round(pos_marker_radius): center_y+noisey[0]+round(pos_marker_radius)+1] = 2

    # ******************** Negative Marker ********************
    neg_marker_index = sample_spherical(center_x, center_y, sphere_radius, marker.shape)
    
    for index in range(len(neg_marker_index[0])):
        marker[neg_marker_index[0][index]-2: neg_marker_index[0][index]+2+1, 
                neg_marker_index[1][index]-2: neg_marker_index[1][index]+2+1] = 1
        
    
    return marker

def dilate_lesion(lesion, sphere_radius, M):
    
    dilate_size = sphere_radius*M
    dilate_size = max(dilate_size, 1)
    
    dilated_lesion = ndimage.grey_dilation(lesion, footprint=np.ones((int(2*dilate_size+1), int(2*dilate_size+1))).astype("uint8"))
    dilated_lesion = np.uint8(dilated_lesion*2) - np.uint8(lesion)
    
    return dilated_lesion
    
