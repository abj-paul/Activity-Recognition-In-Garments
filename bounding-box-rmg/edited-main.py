import cv2
import numpy as np

# Load the YOLOv3 model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Get the output layer names
layer_names = net.getLayerNames()

# Get the names of the output layers
#layer_names = net.getUnconnectedOutLayers()
#print(net.getUnconnectedOutLayers())

# Convert the output layer names to strings
output_layers = [str(layer_name) for layer_name in layer_names]

# Print the output layer names
print(output_layers)

# Load the image
image = cv2.imread('chinese-line.jpg')

# Preprocess the image
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

# Set the input blob for the network
net.setInput(blob)

# Forward pass through the network
outs = net.forward(output_layers)

'''
for out in outs:
    for detection in out:
        print(out)
'''
# Get the bounding boxes, confidences, and class IDs
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
            print("Exception occurred!")
            continue

print(boxes)

# Apply non-maximum suppression to remove overlapping bounding boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw the bounding boxes and labels on the image
colors = np.random.uniform(0, 255, size=(len(class_ids), 3))
if len(indices) > 0:
    for i in indices.flatten():
        x, y, width, height = boxes[i]
        label = str(class_ids[i])
        confidence = confidences[i]
        color = colors[i]
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image with the bounding boxes
'''
cv2.imshow('Image with bounding boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
cv2.imwrite('output.png', image)

