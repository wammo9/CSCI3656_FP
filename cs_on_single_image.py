import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

image = cv2.imread("/Users/elizabethmcguire/Desktop/CSCI3656_FP/valid_mouths1/478smile.jpg")

# Crop everything but the mouth
# Luckily the mouths are all the same place in the dataset 
x_center, y_center = 30, 30
crop_size_x = 40
crop_size_y = 30
x1 = max(0, x_center - crop_size_x // 2)
y1 = max(0, y_center - crop_size_y // 2)
x2 = min(60, x_center + crop_size_x // 2)
y2 = min(60, y_center + crop_size_y // 2)

crop = image[y1:y2, x1:x2]

# Here is where the image processing starts
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

# Blur and Canny
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 30, 100)

plt.imshow(edges, cmap='gray')
plt.title("Canny Edges")
plt.axis('off')
plt.show()

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
temp = crop.copy()
cv2.drawContours(temp, contours, -1, (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
plt.title("Detected Contours")
plt.axis('off')
plt.show()

filtered_contours = [c for c in contours if cv2.contourArea(c) > 10]
# image processing ends

# take the points that will be used to fits the spline
if filtered_contours:
    # take the largest contour
    contour = max(filtered_contours, key=cv2.contourArea)
    # outline the contour
    hull = cv2.convexHull(contour)
    mouth_points = hull[:, 0, :]

    sorted_by_y = mouth_points[mouth_points[:, 1].argsort()]
    # after marking points this keeps only the bottom 70% of points
    # its most of the points because the bottom lip is the biggest contour anyways
    bottom_lip = sorted_by_y[int(0.1 * len(sorted_by_y)):]
    # sort the points for the interpolation
    bottom_lip = bottom_lip[bottom_lip[:, 0].argsort()]

    x = bottom_lip[:, 0]
    y = bottom_lip[:, 1]

    # Remove duplicate x-values by keeping only the first occurrence
    # cubic spline interpolation can't have duplicate or decreasing x
    unique_x, unique_indices = np.unique(x, return_index=True)
    unique_y = y[unique_indices]

    if len(unique_x) > 3:
        spline = CubicSpline(unique_x, unique_y)
        x_new = np.linspace(unique_x[0], unique_x[-1], 100)
        y_new = spline(x_new)
    else:
        print("Not enough unique x-values to fit a spline.")

    plt.scatter(x, y, color='red', label="Bottom Lip Points")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title("Extracted Bottom Lip Points")
    plt.show()

    # Visualize points directly on the cropped image
    overlay = crop.copy()
    for (px, py) in zip(x, y):
        cv2.circle(overlay, (int(px), int(py)), radius=2, color=(0, 0, 255), thickness=-1)

    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Bottom Lip Points on Cropped Image")
    plt.axis('off')
    plt.show()

    if len(unique_x) > 3:
        # fit the cubic spline to the x and y values found above
        cs = CubicSpline(unique_x, unique_y)
        x_fine = np.linspace(unique_x[0], unique_x[-1], 100)
        y_fine = cs(x_fine)
        second_deriv = cs.derivative(2)(x_fine)
        # print(f"Second Derivative: {second_deriv}")

        # A big max means an upward looking thing
        max_curvature = np.max(second_deriv)
        # A big min means more downward
        min_curvature = np.min(second_deriv)
        # Just taking the max and min means that is where the 
        # curvature is sharpest which can lead to an overall
        # mininterpretation of the curve so we need the average of
        # these values to smooth it out
        avg_curvature = (max_curvature + min_curvature) / 2
        print(f"Max Curvature: {max_curvature}, Min Curvature: {min_curvature}")

        if avg_curvature > 0:
            expression = "Smile :)"
        elif avg_curvature < 0:
            expression = "Frown :("

        plt.scatter(unique_x, unique_y, color='red', label="Original Points")
        plt.plot(x_fine, y_fine, color='blue', label="Cubic Spline Fit")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.title("Cubic Spline Fit to Bottom Lip")
        plt.show()

        print(f"Detected Expression: {expression}")
    else:
        print("Not enough unique x-values to fit a spline.")
else:
    print("No suitable mouth contour found.")
