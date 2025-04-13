# app.py

import streamlit as st
from PIL import Image
import tempfile
import numpy as np
import cv2 as cv
from cv2 import cvtColor,resize
import matplotlib.pyplot as plt
import webbrowser

import KnnColorFunctions as Knnfcn


st.title("🖼️ Simple Image Transformer")

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
