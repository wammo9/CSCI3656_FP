# Go through all the images in the dataset 
# and copy images that can actually be processed 
# to new folder

import os
import cv2
import numpy as np
import shutil

# dataset folder and new folder with valid images
data = "/Users/elizabethmcguire/Desktop/CSCI3656_FP/mouth"
new_data = "/Users/elizabethmcguire/Desktop/CSCI3656_FP/valid_mouths2"

# Create folder if it doesn't already exist
os.makedirs(new_data, exist_ok=True)

# Function to process a single image and return whether it's valid
def is_valid_mouth_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return False

    x_center, y_center = 30, 30
    crop_size_x = 40
    crop_size_y = 30
    x1 = max(0, x_center - crop_size_x // 2)
    y1 = max(0, y_center - crop_size_y // 2)
    x2 = min(60, x_center + crop_size_x // 2)
    y2 = min(60, y_center + crop_size_y // 2)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [c for c in contours if cv2.contourArea(c) > 10]

    if not filtered_contours:
        return False

    contour = max(filtered_contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    mouth_points = hull[:, 0, :]
    sorted_by_y = mouth_points[mouth_points[:, 1].argsort()]
    bottom_lip = sorted_by_y[int(0.3 * len(sorted_by_y)):]
    bottom_lip = bottom_lip[bottom_lip[:, 0].argsort()]
    x = bottom_lip[:, 0]
    y = bottom_lip[:, 1]
    unique_x, unique_indices = np.unique(x, return_index=True)
    unique_y = y[unique_indices]

    return len(unique_x) > 3

# Loop through all images in the dataset
for root, _, files in os.walk(data):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(root, file)
            try:
                if is_valid_mouth_image(image_path):
                    # copy to new folder
                    shutil.copy(image_path, os.path.join(new_data, file))
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

