'''
Document field detection using Template Matching
'''
# Import necessary libraries
import numpy as np
import cv2

# Updated threshold values for different fields
field_threshold = {
    "prev_policy_no": 0.8,  # Updated threshold value for prev_policy_no
    "address": 0.7,        # Updated threshold value for address
}

# Function to generate bounding boxes around detected fields
def draw_bounding_boxes(image, gray_image, template, field_name="field"):
    # Get the dimensions of the template
    width, height = template.shape[::-1]

    # Apply template matching using normalized correlation coefficient method
    result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)

    # Find locations where the match score is above the threshold
    locations = np.where(result >= field_threshold[field_name])

    # Draw rectangles around the detected regions
    for point in zip(*locations[::-1]):
        top_left = point
        bottom_right = (top_left[0] + width, top_left[1] + height)
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 255), 2)  # Yellow rectangles

        # Add text label near the detected region
        text_y = top_left[1] - 10 if top_left[1] - 10 > 10 else top_left[1] + height + 20
        cv2.putText(image, field_name, (top_left[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)  # Red text

    return image

# Main function to execute the field detection
if __name__ == '__main__':
    # Load the main document image
    document_image = cv2.imread('document_sample.png')  # Updated image file name

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(document_image, cv2.COLOR_BGR2GRAY)

    # Load the templates for different fields
    template_address = cv2.imread('template_address.png', 0)  # Updated template file name
    template_policy = cv2.imread('template_prev_policy.png', 0)  # Updated template file name

    # Detect and draw bounding boxes for each field
    document_image = draw_bounding_boxes(document_image.copy(), gray_image.copy(), template_address, 'address')
    document_image = draw_bounding_boxes(document_image.copy(), gray_image.copy(), template_policy, 'prev_policy_no')

    # Display the image with detected fields
    cv2.imshow('Detected Fields', document_image)  # Window title
    cv2.waitKey(0)
    cv2.destroyAllWindows()
