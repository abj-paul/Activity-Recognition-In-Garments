import cv2
import numpy as np


def get_bounding_box_for_image(input_image_path, output_image_path):
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    layer_names = net.getLayerNames()
    output_layers = [str(layer_name) for layer_name in layer_names]

    image = cv2.imread(input_image_path)
    # Preprocess the image
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Forward pass through the network
    outs = net.forward(output_layers)

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

    cv2.imwrite(output_image_path, image)

'''
inp = 'image.jpg'
#inp = 'chinese-line.jpg'
out = 'output.png'
get_bounding_box_for_image(inp, out)
'''

