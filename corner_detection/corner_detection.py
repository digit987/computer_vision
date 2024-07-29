'''
Using Shi-Tomasi method to detect corners
'''
# Import the necessary libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('corner.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect corners using the goodFeaturesToTrack function
# New parameters:
# - Maximum corners: 50
# - Quality level: 0.02
# - Minimum distance between corners: 15
detected_corners = cv2.goodFeaturesToTrack(gray_image, 50, 0.02, 15)

# Convert the corners to integer values
detected_corners = np.int0(detected_corners)

# Iterate through each detected corner
# Draw a circle with new parameters:
# - Radius: 5
# - Color: black (0)
# - Thickness: 2 (outline)
for corner in detected_corners:
    x, y = corner.ravel()
    cv2.circle(image, (x, y), 5, 0, 2)

# Display the image with the detected corners
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying with Matplotlib
plt.title('Corners Detected')
plt.show()
