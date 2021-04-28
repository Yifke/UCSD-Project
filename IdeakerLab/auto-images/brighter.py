import cv2
import matplotlib.pyplot as plt
  
if __name__ == "__main__":

    mylist = range(1,190)
    for ind in mylist:
        mImage = cv2.imread('../Images/zstack1/zstack1z%03d.tif'% (ind))
        hsvImg = cv2.cvtColor(mImage,cv2.COLOR_BGR2HSV)
        # decreasing the V channel by a factor from the original
        hsvImg[...,2] = hsvImg[...,2]*5
        new_img = cv2.cvtColor(hsvImg,cv2.COLOR_HSV2RGB)
        cv2.imwrite('./sample_images/5/sample%03d.png'% (ind), new_img)