import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load the image
image = cv2.imread('captured_image.jpg')

# Reshape image to a 2D array of pixels
image_reshaped = image.reshape((-1, 3))

# Perform K-means clustering to find dominant color
kmeans = KMeans(n_clusters=1)
kmeans.fit(image_reshaped)

# Get the dominant color (centroid)
dominant_color = kmeans.cluster_centers_[0]
print(f"Dominant color (RGB): {dominant_color}")
