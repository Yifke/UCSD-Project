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

#This block imports the scripts from the src folder

def get_path(index):
    """Generate the path of the image

    Parameters
    ----------
    index : int
        The index of the image to read

    Output
    ------
    path : string
        The path of the image to read
    """

    # path = '../data/tube' + str(index) + '.tif'
    # path = './sample_images/4/sample%03d.png'% (index)
    # path = './test_image/sample%03d.png'% (index)
    path = './inverse_images/4/sample%03d.png'% (index)

    return path

# @profile
def preprocess(img):
    """Threshold an image using grayscale and global threshold

    Parameters
    ----------
    img : Image
        The image to be grayscaled and thresholded

    Output
    ------
    threshold : Image
        The thresholded image of the input image
    """

    grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    index = max_index(grayscaled)
    retval, threshold = cv2.threshold(grayscaled, index, 255, cv2.THRESH_BINARY)

    return threshold

def image_denoise(img):
    """Compress the image to 256x256 and denoise the image

    Parameters 
    ----------
    img : Image 
        An image file that will be processed through this function.
        Image can be .tif or .jpg files

    Output
    -------
    img_filtered : Image
        An image that is 256x256 sized and denoised.
    """
    new_img = img_as_float(img)
    new_img = resize(new_img, (int(new_img.shape[0]/4), int(new_img.shape[1]/4)))
    new_img = denoise_tv_chambolle(new_img)
    
    return new_img

def path_Marching_Squares(img_filtered, level):
    """Find contours in input image using Marching Squares algorithm

    Parameters 
    ----------
    img_filtered : Image 
        Input image file in which to find contours.
    level : int
        Value along which to find contours in the array.

    Output
    -------
    contours : list of (n,2)-ndarrays
        Each contour is an ndarray of shape (n, 2), consisting of n (x, y) coordinates along the contour.
    """

    contours = find_contours(img_filtered, level)
    return contours

# @profile
def max_index(img_filtered):
    """Find the level index that can yield the contour with largest area

    Parameters 
    ----------
    img_filtered : Image 
        Input image file in which to find contours.

    Output
    -------
    max_index : int
        Level index that can yield the contour with largest area
    """

    max_index = 0
    max_area = 0
    #max_count = 0

    # 100 - 110
    for i in range(120, 130):
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
    """Find contours in input image using Douglas-Peuker algorithm

    Parameters 
    ----------
    img_filtered : Image 
        Input image file in which to find contours.
    level : int
        Value along which to find contours in the array.
    contours : list of (n,2)-ndarrays
        Each contour is an ndarray of shape (n, 2), consisting of n (x, y) coordinates along the contour.

    Output
    -------
    storage : 2D array of float coordinates
        Output array of detected contour coordinates.
    """

    storage = []
    for contour in contours:
        coords = approximate_polygon(contour, tolerance=0.1)    
        #if isClosed(coords) and polygon_area(coords) > 100:
        if isClosed(coords) and polygon_area(coords) > 100:
            storage.append(coords)
    return storage

def get_centroid(points):
    """Calculate the center of mass of input polygon

    Parameters
    ----------
    points : 2D array
        The vertices coordinate of a polygon

    Output
    ------
        The (x, y) coordinate of the centroid of input polygon
    """

    sum_x = 0
    sum_y = 0
    length = len(points)
    for i in range(0, length):
        sum_x += points[i][0]
        sum_y += points[i][1]
    return (sum_x / length, sum_y / length)

def drawContour(img_num, path, center_list):
    img = cv2.imread(path)
    threshold = preprocess(img)
    sk_img = image_denoise(img)
    sk_thre = image_denoise(threshold)
    contours = path_Marching_Squares(sk_thre, 0.45)

    fig, ax = plt.subplots(figsize=(10,10))
    plt.gray()
    ax.imshow(sk_img)

    for contour in contours:
        coords = approximate_polygon(contour, tolerance=0.1)    
        #if isClosed(coords) and polygon_area(coords) > 100:
        if isClosed(coords) and polygon_area(coords) > 100:
            ax.plot(coords[:, 1], coords[:, 0], '-r', linewidth=2)
    for center in center_list:
        ax.scatter(center[1], center[0], marker='o')
    ax.axis((0,256,0,256))
    # save_path = '../data/auto/segmented_tube' + str(img_num) + '.jpg'
    # save_path = './sample_results/4/sample%03d.png'% (img_num)
    # save_path = './test_result/sample%03d.png'% (img_num)
    save_path = './inverse_result/sample%03d.png'% (img_num)
    plt.savefig(save_path)