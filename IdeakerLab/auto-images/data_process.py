import numpy as np
from MyCluster import MyCluster
from scipy.spatial import distance
from shapely.geometry import Polygon

poly_list = np.load('poly.npy')
center_list = np.load('center.npy')
brights_list = np.load('brightness.npy')
cluster_list = []

def center_exist(poly_center, poly, brightness, layer_ind):
    for cluster in cluster_list:
        dis = distance.euclidean(poly_center, cluster.center[-1])
        # Distance between centers
        if dis <= 10:
            if cluster.image_list[-1] != layer_ind:
                cluster.image_list.append(layer_ind)
                cluster.center.append(poly_center)
                cluster.brightness_list.append(brightness)
                cluster.contour.append(poly)
            elif cluster.image_list[-1] == layer_ind:
                poly1 = Polygon(poly)
                poly2 = Polygon(cluster.contour[-1])
                if poly1.area > poly2.area:
                    cluster.center[-1] = poly_center
                    cluster.contour[-1] = poly
                    cluster.brightness_list[-1] = brightness
            return True
        elif dis <= 50:
            poly1 = Polygon(poly)
            poly2 = Polygon(cluster.contour[-1])
            if poly1.is_valid and poly2.is_valid:
                intersection = poly1.intersection(poly2)
                if intersection.area / poly1.area >= 0.8 or intersection.area / poly2.area >= 0.8:
                    if cluster.image_list[-1] != layer_ind:
                        cluster.image_list.append(layer_ind)
                        cluster.center.append(poly_center)
                        cluster.contour.append(poly)
                        cluster.brightness_list.append(brightness)
                    elif cluster.image_list[-1] == layer_ind:
                        if poly1.area > poly2.area:
                            cluster.center[-1] = poly_center
                            cluster.contour[-1] = poly
                            cluster.brightness_list[-1] = brightness
                    return True
    return False

if __name__ == "__main__":
    mylist = range(1, 190)
    for ind in mylist:
        centers = center_list[ind-1]
        shapes = poly_list[ind-1]
        brights = brights_list[ind-1]
        length = len(centers)
        for i in range(0, length):
            cent = centers[i]
            cont = np.asarray(shapes[i])
            brightness = brights[i]
            if(center_exist(cent, cont, brightness, ind) == False):
                cluster = MyCluster()
                cluster.center.append(cent)
                cluster.contour.append(cont)
                cluster.brightness_list.append(brightness)
                cluster.image_list.append(ind)
                cluster_list.append(cluster)

    print('Number of distinct contours found: %d'% len(cluster_list))
    my_clu = cluster_list[0]
    for i in range(0, 50):
        print('Image%02d: Contour center: (%05f, %05f), Average brightness: %05f' 
            %((my_clu.image_list[i]), my_clu.center[i][1], my_clu.center[i][0], my_clu.brightness_list[i]))
    # for cluster in cluster_list:
    #     print("Cluster Images list: ")
    #     print(cluster.image_list)
    #     print("Cluster Brightness list: ")
    #     print(cluster.brightness_list)
    #     print()
    
    np.save('sample_center', cluster_list[0].center)
    np.save('sample_contour', cluster_list[0].contour)
    np.save('sample_imglist', cluster_list[0].image_list)
    np.save('sample_brightlist', cluster_list[0].brightness_list)


