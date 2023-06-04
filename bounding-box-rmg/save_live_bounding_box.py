from modular_bounding_box import *
from save_bounding_box import *

import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import numpy
import matplotlib.pyplot as mtp
import os
from sklearn.decomposition import NMF
'''
inp = 'image.jpg'
out = 'output.png'
'''
import time

def extractFrameFromImage(filePath, imageLimit, frame_rate):
  vidcap = cv2.VideoCapture(filePath)
  success,image = vidcap.read()
  count = 0
  #define framerate
  prev = 0
  frame=0
  while success:
    time_elapsed = time.time() - prev
    success,image = vidcap.read()  
    if time_elapsed > 1./frame_rate:
      prev = time.time()
      cv2.imwrite("%d.jpg" % count, image)     # save frame as JPEG file      
      print('Read a new frame: ', count, success)
      count=count+1
      frame += frame_rate # i.e. at 30 fps, this advances one second
      vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    if count>=imageLimit:
      break
  vidcap.release()
  #cv2.destroyAllWindows()

extractFrameFromImage(filePath="video.mp4", imageLimit = 50, frame_rate=5)
#extractFrameFromImage(filePath="http://10.100.103.245:4747/video", imageLimit = 20, frame_rate=5)
#extractFrameFromImage(filePath="Abhijit-activity.mp4", imageLimit = 20, frame_rate=5)
for count in range(20):
    input_img_path = str(count)+".jpg"
    output_img_path = "frame_"+str(count)
    save_bounding_box_images(input_img_path, output_img_path)
