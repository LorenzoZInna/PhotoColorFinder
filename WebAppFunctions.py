import streamlit as st
from streamlit_clickable_images import clickable_images
import base64
import os

def get_base64_image(path):
    with open(path, "rb") as img_file:
        return "data:image/png;base64," + base64.b64encode(img_file.read()).decode()

def return_urls():
    # List of local image paths
    image_folder = "images"
    image_files = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.endswith(".png")
    ])

    # Convert local images to base64 data URLs
    image_data_urls = [get_base64_image(img_path) for img_path in image_files]
    return image_data_urls

def delete_images():

    # Specify the folder path
    folder_path = 'images'

    # Loop through the files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):  # Check for PNG images (case insensitive)
            file_path = os.path.join(folder_path, filename)  # Full path of the file
            try:
                os.remove(file_path)  # Delete the file
                print(f'Deleted: {file_path}')
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
