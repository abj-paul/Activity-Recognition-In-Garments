import cv2
import numpy as np

# Load the model
model = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
print(model)
# Detect humans in the image
image = cv2.imread('image.jpg')
print(image.shape)
# Get the bounding boxes of the detected humans
boxes = model.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw the bounding boxes on the image
for box in boxes:
    x, y, w, h = box
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the image with the bounding boxes
cv2.imshow('Image with bounding boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

