# This will be cubic splines run on all the images
# that can be processed. And it will determine how well
# this method works because we will check how many
# it gets right as in how many smiles are actually smiles
# how many frowns are actually frowns.

import os
import cv2
import numpy as np
from scipy.interpolate import CubicSpline

images = "/Users/elizabethmcguire/Desktop/CSCI3656_FP/valid_mouths2"

smile_count = 0
frown_count = 0

# Loop through every image in the folder
for filename in os.listdir(images):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        filepath = os.path.join(images, filename)
        image = cv2.imread(filepath)

        # The following is the same code from
        # single image file so all the same from here
        x_center, y_center = 30, 30
        crop_size_x = 40
        crop_size_y = 30
        x1 = max(0, x_center - crop_size_x // 2)
        y1 = max(0, y_center - crop_size_y // 2)
        x2 = min(60, x_center + crop_size_x // 2)
        y2 = min(60, y_center + crop_size_y // 2)

        crop = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [c for c in contours if cv2.contourArea(c) > 10]

        if filtered_contours:
            contour = max(filtered_contours, key=cv2.contourArea)
            hull = cv2.convexHull(contour)
            mouth_points = hull[:, 0, :]

            sorted_by_y = mouth_points[mouth_points[:, 1].argsort()]
            bottom_lip = sorted_by_y[int(0.1 * len(sorted_by_y)):]
            bottom_lip = bottom_lip[bottom_lip[:, 0].argsort()]
            x = bottom_lip[:, 0]
            y = bottom_lip[:, 1]

            unique_x, unique_indices = np.unique(x, return_index=True)
            unique_y = y[unique_indices]

            if len(unique_x) > 3:
                cs = CubicSpline(unique_x, unique_y)
                x_fine = np.linspace(unique_x[0], unique_x[-1], 100)
                y_fine = cs(x_fine)
                second_deriv = cs.derivative(2)(x_fine)
                max_curvature = np.max(second_deriv)
                min_curvature = np.min(second_deriv)
                avg_curvature = (max_curvature + min_curvature) / 2

                # Increment the counters
                if avg_curvature > 0:
                    smile_count += 1
                elif avg_curvature < 0:
                    frown_count += 1

print(f"\nTotal Smiles: {smile_count}")
print(f"Total Frowns: {frown_count}")

