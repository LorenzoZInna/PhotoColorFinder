import cv2

# Initialize the camera
camera = cv2.VideoCapture(0)

# Capture a frame from the camera
ret, frame = camera.read()

# Save the image
if ret:
    cv2.imwrite("captured_image.jpg", frame)

camera.release()
cv2.destroyAllWindows()
