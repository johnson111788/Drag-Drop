import cv2
import numpy as np
from skimage.filters import  rank


def load_img_label(imgpath, labelpath):
    
    label = cv2.imread(labelpath)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY) // 255
    
    image = cv2.imread(imgpath)
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return grayimage, label, image


def cal_img_gradient(img):
    
    img = rank.median(img, np.ones((2, 2)))
    gradient_volume = rank.gradient(img, np.ones((2, 2)))
    
    return gradient_volume


def save_output(file_path, pred, img, pred_color = [244, 133, 0], mask_color = [255, 179, 255]):

    pred_gt = cv2.cvtColor(np.uint8(pred == 1),cv2.COLOR_GRAY2RGB)
    pred_mask = cv2.cvtColor(np.uint8(pred == 2),cv2.COLOR_GRAY2RGB)
    pred = pred_gt * pred_color + pred_mask * mask_color

    img_pred = cv2.addWeighted(np.uint8(img), 0.9, np.uint8(pred), 0.6, 1)
    
    cv2.imwrite(file_path, img_pred)
    

def count_integer_points_on_sphere(radius):

    x, y= np.meshgrid(range(-radius, radius + 1), range(-radius, radius + 1))
    distances_squared = x**2 + y**2
    count = np.count_nonzero(abs(distances_squared - radius**2) <= 1)

    return count


def sample_spherical(cx, cy, radius, shape):

    num_points = count_integer_points_on_sphere(radius)
    
    t = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    x = np.int32(cx + radius * np.cos(t))
    y = np.int32(cy + radius * np.sin(t))
    
    spherical_index = np.array([x,y])
    
    spherical_index[0][spherical_index[0]+2+1>=shape[0]] = shape[0]-1-2
    spherical_index[1][spherical_index[1]+2+1>=shape[1]] = shape[1]-1-2
    spherical_index[spherical_index<0] = 0
    
    return spherical_index

