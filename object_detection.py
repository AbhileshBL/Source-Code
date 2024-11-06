import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()

# Adjusting for different versions of OpenCV
unconnected_layers = net.getUnconnectedOutLayers()
if isinstance(unconnected_layers[0], np.ndarray):
    # Newer OpenCV versions return each layer as a scalar, so access by index directly
    output_layers = [layer_names[i[0] - 1] for i in unconnected_layers]
else:
    # Older OpenCV versions return integer layer IDs directly
    output_layers = [layer_names[i - 1] for i in unconnected_layers]

# Load input image
img = cv2.imread("input.jpg")
height, width, channels = img.shape

# Prepare image for YOLO
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# Run inference
outs = net.forward(output_layers)

# Processing results
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-max suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw bounding boxes
for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = str(class_ids[i])
    confidence = confidences[i]
    color = (0, 255, 0)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
