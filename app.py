# app.py

import streamlit as st
from PIL import Image
import tempfile
import numpy as np
import cv2 as cv
from cv2 import cvtColor,resize
import matplotlib.pyplot as plt
import webbrowser
import WebAppFunctions

import streamlit as st
from streamlit_clickable_images import clickable_images

import KnnColorFunctions as Knnfcn


st.title("🖼️ KNN Color Finder")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)
    image.save("data/input.jpg")  # Save as JPEG
    st.image(image, caption="Uploaded Image", use_container_width =True)
    reshaped_image=Knnfcn.reshape_image("data/input.jpg")
    decomposed_reshaped_image_3dplot = Knnfcn.reshaped_image_3d_plot("data/input.jpg")
    # Convert to grayscale (simple transformation)
    st.subheader("Decomposed Colors from Image")
    st.pyplot(decomposed_reshaped_image_3dplot, use_container_width=True)
    st.caption("All pixels of the intial image are decomposed into their RGB components and represented in a 3D plot")
    k_index,centroids_k,labels_k,k_inertia,silhouette_k = Knnfcn.KMeans_models(reshaped_image)
    red_array,green_array,blue_array = Knnfcn.colors_k_centroids_all (centroids_k)
    centroid_k_main_index,percentage_counts_k,k_main_color_list = Knnfcn.three_main_centroids_colors(labels_k,red_array,green_array,blue_array)
    Knnfcn.create_color_rectangle_centroid(k_main_color_list,percentage_counts_k)

    import streamlit as st
    import os

    image_folder = "images"
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])

    st.write("Choose an image:")

    for i, filename in enumerate(image_files):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(os.path.join(image_folder, filename), width=100)
        with col2:
            if st.button(f"Select {filename}", key=filename):
                st.write(f"You selected: {filename}")
                st.image(os.path.join(image_folder, filename), caption=filename, use_container_width=True)
