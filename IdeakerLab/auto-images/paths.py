import numpy as np

def dist(x,y): 
    return np.sqrt(np.power(x - y, 2).sum())

def isClosed(path, tol= 0.001): 
    if dist(path[0], path[-1]) < tol: 
        return True

    return False

def polygon_area(path): 
    assert isClosed(path)
    
    area = 0
    for ind, i in enumerate(path[:-1]): 
        area += np.linalg.det(path[ind:ind+2, :])
        
    area += np.linalg.det(path[[0,-1], :])
    
    return area/2

##TODO: def centroid(path)