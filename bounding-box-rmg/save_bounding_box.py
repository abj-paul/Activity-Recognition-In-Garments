import cv2
import numpy as np

def save_bounding_box_images(input_image_path, output_image_path):
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
                continue

    # Apply non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Save each bounding box as a separate image
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, width, height = boxes[i]
            roi = image[y:y+height, x:x+width]
            output_image_path = f'{output_image_path}_bounding_box_{i}.jpg'
            cv2.imwrite(output_image_path, roi)

# Example usage
'''
input_image_path = 'image.jpg'
output_image_dir = 'bounding_box_images'
save_bounding_box_images(input_image_path, output_image_dir)
'''

