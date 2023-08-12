import cv2
import os
import numpy as np
from PIL import Image
import imagehash

# Load the object detection model (YOLO)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")
output_layers = net.getUnconnectedOutLayersNames()

# Supported image file extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp",
                        ".svg", ".raw", ".heic", ".heif", ".ico", ".jp2", ".exr",
                        ".pbm", ".pgm", ".ppm", ".jxr"}

# Create a dictionary to store image hashes and their filenames
image_database = {}

# Populate the image database with hashes
def build_image_database(directory):
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            img_hash = imagehash.average_hash(img)
            image_database[img_hash] = filename

# Find similar images based on hash similarity
def find_similar_images(query_img_path, threshold=10):
    query_img = Image.open(query_img_path)
    query_hash = imagehash.average_hash(query_img)
    
    similar_images = []
    for img_hash, filename in image_database.items():
        if abs(query_hash - img_hash) <= threshold:
            similar_images.append(filename)
    
    return similar_images

# Perform object detection on an image
def detect_objects(image_path):
    img = cv2.imread(image_path)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    detected_objects = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Adjust confidence threshold as needed
                label = classes[class_id]
                detected_objects.append(label)
    
    return detected_objects

# Search for individual object matches in the image database
def search_database_for_objects(detected_objects):
    matching_images = {}
    for obj in detected_objects:
        print(f"Searching for matches of '{obj}' in the database...")
        similar_images = find_similar_images(object_image_path)
        matching_images[obj] = similar_images
    return matching_images

# Example usage
build_image_database("/home/durai/Documents/code/image_data")  # Replace with the path to your image directory
object_image_path = "elon22.jpg"  # Replace with the path to your query object image

# Detect objects in the query image
detected_objects = detect_objects(object_image_path)
print("Detected objects:", detected_objects)

# Search for object matches in the image database
matching_images = search_database_for_objects(detected_objects)

# Print matching images for each detected object
for obj, similar_images in matching_images.items():
    print(f"Similar images containing '{obj}':")
    for image in similar_images:
        print(image)
