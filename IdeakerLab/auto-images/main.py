import os
import cv2
import numpy as np
import MyImage
import multiprocessing 
# from MyContour import MyContour
from matplotlib import pyplot as plt
from multiprocessing import Pool
from scipy.spatial import distance_matrix
from line_profiler import LineProfiler
import random

layer_polys = []
layer_centers = []
layer_brightness = []
# @profile
def process_image(ind):
    center_list = []
    brightness_list = []
    img_path = MyImage.get_path(ind)
    img = cv2.imread(img_path)
    threshold = MyImage.preprocess(img)

    sk_img = MyImage.image_denoise(img)
    sk_thre = MyImage.image_denoise(threshold)
    contour = MyImage.path_Marching_Squares(sk_thre, 0.45)
    contours = MyImage.path_tracing(sk_img, contour)

    for poly in contours:
        poly_center = MyImage.get_centroid(poly)
        center_list.append(poly_center)

        mask = np.zeros_like(sk_img)
        cv2.drawContours(mask, [poly.astype(int)], -1, color=255, thickness=-1)
        points = np.where(mask == 255)
        total_brightness = 0
        for point in points:
            R = sk_img[point[1], point[0]][0]
            G = sk_img[point[1], point[0]][1]
            B = sk_img[point[1], point[0]][2]
            total_brightness += (0.2126*R + 0.7152*G + 0.0722*B)
        brightness = total_brightness/len(points)
        brightness_list.append(brightness)
    layer_polys.append(np.array(contours))
    layer_centers.append(np.array(center_list))
    layer_brightness.append(np.array(brightness_list))
    MyImage.drawContour(ind, img_path, center_list)

if __name__ == "__main__":
    mylist = range(1,190)
    for ind in mylist:
        p = multiprocessing.Process(process_image(ind))
        p.start()
    p.join()

    np.save('poly', layer_polys)
    np.save('center', layer_centers)
    np.save('brightness', layer_brightness)
    

