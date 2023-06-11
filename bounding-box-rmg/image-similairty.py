import os
import skimage
import cv2
import numpy as np
import numpy
import os
import time
import random

# Objects to track
class Tracker:
    __isInitialized = False
    __oldState = []

    def __init__(self,rootDir):
        self.rootDir = rootDir
        if not os.path.exists(self.rootDir):
            os.mkdir(self.rootDir)


    def __initialize(self, MAX_NUM_OF_OBJECTS, SAMPLE_OBJECT):
        for objecT_index in range(MAX_NUM_OF_OBJECTS):
            self.__oldState.append(SAMPLE_OBJECT)
        self.__isInitialized = True

    def trackObject(self, current_object_state):
        '''
        if not self.__isInitialized:
            self.__initialize(1, current_object_state)
        '''

        for object_index, old_object_state in enumerate(self.__oldState):
            if self.calculate_image_similarity(old_object_state, current_object_state)==1:
                self.__oldState[object_index] = current_object_state
                return "Object"+str(object_index)

        # If the object is not registered previously, then register it.
        lastIndex = len(self.__oldState)
        self.__oldState.append(current_object_state)
        folder_name = "Object"+str(lastIndex)
        if not os.path.exists(f'{self.rootDir}/{folder_name}'):
            os.mkdir(f'{self.rootDir}/{folder_name}')
        return folder_name
    

    def calculate_image_similarity(self, image1, image2):
         return random.randint(0, 1)
    
class RMGSystem:
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    frameDir = "frames"
    rootDir = "./data"
    tracker = Tracker(rootDir)

    def __init__(self):
        if not os.path.exists(self.rootDir):
            os.mkdir(self.rootDir)
        dirPath = self.rootDir + "/" + self.frameDir
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)

    def extractFrameFromVideoStream(self, videoStream, imageLimit, frame_rate):
      vidcap = cv2.VideoCapture(videoStream)
      success,image = vidcap.read()
      count = 0
      #define framerate
      frame=0
      prev = 0
      while success:
        time_elapsed = time.time() - prev
        success,image = vidcap.read()  
        if time_elapsed > 1./frame_rate:
          prev = time.time()
          cv2.imwrite(f"{self.rootDir+'/'+self.frameDir}/frame{count}.jpg", image)     # save frame as JPEG file      
          print('Read a new frame: ', count, success)
          count=count+1
          frame += frame_rate # i.e. at 30 fps, this advances one second
          vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        if count>=imageLimit:
          break
      vidcap.release()
      #cv2.destroyAllWindows()



    def save_bounding_box_images(self, input_image_path):
        layer_names = self.net.getLayerNames()
        output_layers = [str(layer_name) for layer_name in layer_names]
    
        image = cv2.imread(input_image_path)
        # Preprocess the image
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
    
        # Forward pass through the network
        outs = self.net.forward(output_layers)
    
        boxes = []
        confidences = []
        class_ids = []
        for out in outs:
            for detection in out:
                detection = np.array(detection)
                try:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:  # Adjust the confidence threshold as needed
                        center_x = int(detection[0] * image.shape[1])
                        center_y = int(detection[1] * image.shape[0])
                        width = int(detection[2] * image.shape[1])
                        height = int(detection[3] * image.shape[0])
                        x = int(center_x - width / 2)
                        y = int(center_y - height / 2)
                        boxes.append([x, y, width, height])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
                except:
                    continue
    
        # Apply non-maximum suppression to remove overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
        # Save each bounding box as a separate image
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, width, height = boxes[i]
                roi = image[y:y+height, x:x+width]
                object_folder = self.tracker.trackObject(roi)
                output_image_path = f'{self.rootDir}/{object_folder}/bounding_box_{i}.jpg'
                cv2.imwrite(output_image_path, roi)
                print(output_image_path + "==>"+str(object_folder))
    def testClass(self):
        self.extractFrameFromVideoStream(videoStream = "video.mp4", imageLimit = 5, frame_rate=5)
        for fileName in os.listdir(f'{self.rootDir}/{self.frameDir}'):
            print(fileName)
            self.save_bounding_box_images(f'{self.rootDir}/{self.frameDir}/{fileName}')

rmgSystem = RMGSystem()
rmgSystem.testClass()

