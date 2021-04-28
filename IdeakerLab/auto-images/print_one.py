import os
import cv2
import numpy as np
from MyImage import image_denoise
import multiprocessing 
from matplotlib import pyplot as plt
from scipy.spatial import distance
from shapely.geometry import Polygon
from skimage.measure import approximate_polygon

layer_center = np.load('sample_center.npy')
layer_poly = np.load('sample_contour.npy')
images = np.load('sample_imglist.npy')
brightness = np.load('sample_brightlist.npy')

for ind in range(0, len(images)):
    path = path = './sample_images/4/sample%03d.png'% (images[ind])
    img = cv2.imread(path)
    sk_img = image_denoise(img)

    fig, ax = plt.subplots(figsize=(10,10))
    plt.gray()
    ax.imshow(sk_img)

    contour = layer_poly[ind]
    center = layer_center[ind]
    brights = brightness[ind]
    coords = approximate_polygon(contour, tolerance=0.1)    
    ax.plot(coords[:, 1], coords[:, 0], '-r', linewidth=2)
    ax.scatter(center[1], center[0], marker='o')
    ax.axis((0,256,0,256))
    save_path = './test_result/sample%03d.png'% (images[ind])
    plt.savefig(save_path)
    print('Image:%3d Brightness:%05f'% ((images[ind]), (brightness[ind])))
