import numpy as np
import cv2
img = cv2.imread('vr.jpeg', cv2.IMREAD_GRAYSCALE)
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img,None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
cv2.imwrite('orb_image.jpg',img2)
