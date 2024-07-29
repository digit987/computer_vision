import cv2
import numpy as np

# Step 1: Read the image
img = cv2.imread('eyes.jpg', cv2.IMREAD_COLOR)

# Step 2: Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 3: Apply blur to the grayscale image using a 3x3 kernel
gray_blurred = cv2.blur(gray, (3, 3))

# Step 4: Apply Hough Circle Transform on the blurred image
detected_circles = cv2.HoughCircles(gray_blurred, 
                                    cv2.HOUGH_GRADIENT, 1, 20, 
                                    param1=50, param2=30, 
                                    minRadius=1, maxRadius=40)

# Step 5: Draw circles on the original image where circles are detected
if detected_circles is not None:
    # Convert the circle parameters (a, b, r) to integers.
    detected_circles = np.uint16(np.around(detected_circles))
    
    # Iterate over all detected circles
    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
        
        # Step 6: Draw the circumference of the circle
        cv2.circle(img, (a, b), r, (0, 255, 0), 2)
        
        # Step 7: Draw a small circle to show the center
        cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
        
        # Step 8: Display the image with detected circles
        cv2.imshow("Detected Circle", img)
        
        # Step 9: Wait for user input to close the window (0 means wait indefinitely)
        cv2.waitKey(0)
