import cv2

# Load the captured image
image = cv2.imread('captured_image.jpg')

# Convert image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range of colors to detect (example for red color)
lower_red = (0, 120, 70)
upper_red = (10, 255, 255)

# Create a mask to isolate the red areas
mask = cv2.inRange(hsv_image, lower_red, upper_red)

# Apply the mask
result = cv2.bitwise_and(image, image, mask=mask)

# Show the result
cv2.imshow("Filtered Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
