import threading
import cv2
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from tkinter import Tk, filedialog

# --- Object Detection Code ---
def run_object_detection():
    # Initialize Tkinter and hide the main window
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    # Open file dialog to select an image file
    img_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not img_path:
        print("No image selected.")
        return "No image selected."

    # Load YOLO model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    
    unconnected_layers = net.getUnconnectedOutLayers()
    if isinstance(unconnected_layers[0], np.ndarray):
        output_layers = [layer_names[i[0] - 1] for i in unconnected_layers]
    else:
        output_layers = [layer_names[i - 1] for i in unconnected_layers]

    # Load COCO class labels
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Load the selected image
    img = cv2.imread(img_path)
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
    detected_objects = []

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
                detected_objects.append(classes[class_id])

    # Apply non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Dictionary to store object counts
    object_counts = {}
    for i in indexes.flatten():
        label = classes[class_ids[i]]
        if label in object_counts:
            object_counts[label] += 1
        else:
            object_counts[label] = 1

    # Generate a natural language description based on detected objects
    description = "In the uploaded image, you can see "
    descriptions = []
    for obj, count in object_counts.items():
        if count == 1:
            descriptions.append(f"a {obj}")
        else:
            descriptions.append(f"{count} {obj}s")
    description += ", ".join(descriptions) + "."

    print("Detected Objects Description:", description)
    return description

# --- Caption Generation Code ---
def generate_caption(prompt, max_length=80):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated Caption:", caption)
    return caption

# Run both object detection and caption generation in sequence
def main():
    # Run object detection and get description
    description = run_object_detection()
    if description == "No image selected.":
        return

    # Use the description as a prompt for caption generation
    prompt = f"The image shows {description}"
    caption = generate_caption(prompt)

    print("\nFinal Image Description:\n", caption)

# Execute the main function
if __name__ == "__main__":
    main()
