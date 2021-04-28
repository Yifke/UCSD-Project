import cv2
import numpy as np
# import MyContour
# from skimage.draw import circle
from skimage import io, img_as_float
from skimage.transform import resize
from skimage.measure import find_contours, approximate_polygon
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt
from paths import isClosed, polygon_area

def preprocess(img):
    grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    index = max_index(grayscaled)
    retval, threshold = cv2.threshold(grayscaled, index, 255, cv2.THRESH_BINARY)

    return threshold

def image_denoise(img):
    new_img = img_as_float(img)
    new_img = resize(new_img, (int(new_img.shape[0]/4), int(new_img.shape[1]/4)))
    new_img = denoise_tv_chambolle(new_img)
    
    return new_img

def path_Marching_Squares(img_filtered, level):
    contours = find_contours(img_filtered, level)
    return contours

def max_index(img_filtered):
    max_index = 0
    max_area = 0
    #max_count = 0

    # 80 - 120
    for i in range(100, 110):
        area = 0
        retval, threshold = cv2.threshold(img_filtered, i, 255, cv2.THRESH_BINARY)

        sk_img = image_denoise(img_filtered)
        sk_thre = image_denoise(threshold)
        contours = path_Marching_Squares(sk_thre, 0.45)

        for contour in contours:
            coords = approximate_polygon(contour, tolerance=1)    
            if isClosed(coords) and polygon_area(coords) > 100:
                area += polygon_area(coords)

        if area > max_area:
            max_area = area
            max_index = i

    return max_index

def path_tracing(img_filtered, contours):
    storage = []
    for contour in contours:
        coords = approximate_polygon(contour, tolerance=0.1)    
        #if isClosed(coords) and polygon_area(coords) > 100:
        if isClosed(coords) and polygon_area(coords) > 100:
            storage.append(coords)
    return storage

def get_centroid(points):
    sum_x = 0
    sum_y = 0
    length = len(points)
    for i in range(0, length):
        sum_x += points[i][0]
        sum_y += points[i][1]
    return (sum_x / length, sum_y / length)

if __name__ == "__main__":
    path = './sample_images/4/sample109.png'
    img = cv2.imread(path)
    threshold = preprocess(img)
    sk_img = image_denoise(img)
    sk_thre = image_denoise(threshold)
    contours = path_Marching_Squares(sk_thre, 0.45)

    fig, ax = plt.subplots(figsize=(10,10))
    plt.gray()
    ax.imshow(sk_img)

    # for contour in contours:
    for i in range(1, 4):
        contour = contours[i]
        coords = approximate_polygon(contour, tolerance=0.1)    
        if isClosed(coords) and polygon_area(coords) > 100:
            ax.plot(coords[:, 1], coords[:, 0], '-r', linewidth=2)
    ax.axis((0,256,0,256))
    plt.show()
    

