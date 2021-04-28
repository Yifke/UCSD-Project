import cv2
import matplotlib.pyplot as plt
  
if __name__ == "__main__":

    mylist = range(1,190)
    for ind in mylist:
        mImage = cv2.imread('./sample_images/4/sample%03d.png'% (ind))
        new_img = cv2.bitwise_not(mImage)
        cv2.imwrite('./inverse_images/4/sample%03d.png'% (ind), new_img)