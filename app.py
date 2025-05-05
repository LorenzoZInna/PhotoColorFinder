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
import os
import streamlit as st
from streamlit_clickable_images import clickable_images
import atexit
import KnnColorFunctions as Knnfcn

st.title("🖼️ KNN Color Finder")




# Initialize session state
if 'ml_done' not in st.session_state:
    WebAppFunctions.delete_images()
    st.session_state.ml_done = False
if 'ml_result' not in st.session_state:
    st.session_state.ml_result = None
if 'last_uploaded_filename' not in st.session_state:
    st.session_state.last_uploaded_filename = None

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    uploaded_filename = uploaded_file.name


    # Check if the uploaded file is new
    if uploaded_filename != st.session_state.last_uploaded_filename:
        # Reset flags because this is a new image
        st.session_state.ml_done = False
        st.session_state.last_uploaded_filename = uploaded_filename

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

        # Run ML processing only once per new file
        with st.spinner("Running model to find the 3 main colors of the image..."):
            WebAppFunctions.delete_images() #delete images only once


            k_index,centroids_k,labels_k,k_inertia,silhouette_k, k_means_log = Knnfcn.KMeans_models(reshaped_image)
            red_array,green_array,blue_array = Knnfcn.colors_k_centroids_all (centroids_k)
            centroid_k_main_index,percentage_counts_k,k_main_color_list = Knnfcn.three_main_centroids_colors(labels_k,red_array,green_array,blue_array)
            st.session_state.k_main_color_list = k_main_color_list
            st.session_state.ml_result = Knnfcn.create_color_rectangle_centroid(k_main_color_list,percentage_counts_k)
            st.session_state.ml_done = True
            st.success("ML Processing Complete")






    elif st.session_state.ml_done:
        # Already processed, just show results
        st.image(os.path.join("data", "input.jpg"), caption="Uploaded Image", use_container_width=True)
        st.subheader("Detected main color - choose one")






image_folder = "images"
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])

if 'selected_index' not in st.session_state:
    st.session_state.selected_index = None
if 'selected_url' not in st.session_state:
    st.session_state.selected_url = None

for i, filename in enumerate(image_files):

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(os.path.join(image_folder, filename), width=100)
    with col2:
        if st.button(f"Select", key=filename):
            st.session_state.selected_index = i
            if 'k_main_color_list' in st.session_state:
                k_main_color_list=st.session_state.k_main_color_list
                # Get color and URL from your functions
                selected_color = k_main_color_list[i][0]
                ral_name = Knnfcn.classify_RAL(selected_color)
                url = Knnfcn.return_product_URL(ral_name[0], ral_name[1])
                st.write(f"Selected color: {ral_name[1]}")  # Optionally display the selected product name
                st.session_state.selected_url = url

if 'selected_product_name' not in st.session_state:
     st.session_state.selected_product_name = None

if st.session_state.selected_url:
    st.subheader("Typical color products - choose one")
    product_list = ["can_spray", "table_varnish", "wall_paint"]

    for i, filename in enumerate(product_list):

        col1 = st.columns([1,3])
        with col1[0]:
                if st.button(filename):
                # Update session state with the selected product name when button is pressed
                    st.session_state.selected_product_name = filename

if st.session_state.selected_product_name:
    st.subheader("🔗 Selected Product URL")
    st.markdown(f"{st.session_state.selected_url}+{st.session_state.selected_product_name}")
