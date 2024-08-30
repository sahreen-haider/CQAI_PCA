
import numpy as np  
# Importing OpenCV for image processing
import cv2  

import streamlit as st 
# Importing PCA from sklearn for image compression
from sklearn.decomposition import PCA  

# Importing os and random for file handling and random selection
import os, random  

# Importing custom utility functions for image list retrieval, image loading, and PCA application
from utils import get_image_list, load_image, apply_pca

st.set_page_config(layout="wide")  # Setting the layout of the Streamlit app to wide mode

# Check if 'random_file' is in session state, initialize it if not
if 'random_file' not in st.session_state:
    st.session_state.random_file = "cat2.jpg"  # Default image for initial display

st.title("Usage of Principal Component Analysis (PCA) in Image compression")  # Display title in the Streamlit app

# Sidebar widget for file upload with a label for accessibility
uploaded_file = st.sidebar.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg', 'tiff', 'bmp'], label_visibility="collapsed")

# Check if a file is uploaded
if uploaded_file is not None:
    st.sidebar.info(f"File uploaded: {uploaded_file.name}")  # Display the uploaded file name in the sidebar
    # Convert the uploaded file to a NumPy array and decode it into an image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Decode the image in color mode
else:
    # If no file is uploaded, allow the user to select a random example image from a directory
    if st.sidebar.button("Try New Example Image!"):
        random_file = random.choice(get_image_list())  # Choose a random image file from the directory
        st.session_state.random_file = random_file  # Update session state with the selected random file

    st.sidebar.info(f"Example file: {st.session_state.random_file}")  # Display the selected example file in the sidebar
    image = load_image(f"Image/{st.session_state.random_file}")  # Load the selected example image using the custom utility function

# Split the image into its blue, green, and red channels
blue, green, red = cv2.split(image)
st.sidebar.write(blue.shape)  # Display the shape of the blue channel in the sidebar

# Perform PCA on the blue channel to calculate explained variance
pca_temp = PCA().fit(blue)
explained_variance_ratio = pca_temp.explained_variance_ratio_  # Get the explained variance ratio for each component

# Calculate cumulative variance for the PCA components
cumulative_variance = np.cumsum(explained_variance_ratio)

# Function to generate a slider label with the number of components and variance information
def slider_label(components):
    return f"Number of PCA Components: {components}, Variance Preserved: {cumulative_variance[components-1]:.2%}"

# Create a slider for selecting the number of PCA components, with the variance information displayed
pca_components = st.slider(slider_label(20), 1, blue.shape[0], 20, format="%d")
st.sidebar.write(f"Variance preserved: {cumulative_variance[pca_components-1]:.2%}")  # Display the variance preserved in the sidebar

# Apply PCA to each channel using the custom utility function
redI = apply_pca(red, pca_components)
greenI = apply_pca(green, pca_components)
blueI = apply_pca(blue, pca_components)

# Reconstruct the image by stacking the processed channels and converting it to an unsigned 8-bit integer format
re_image = (np.dstack((blueI, greenI, redI))).astype(np.uint8)

# Create two columns to display the original and compressed images side by side
col1, col2 = st.columns(2)
with col1:
    st.header("Original Image")  # Header for the original image
    # Convert the image from BGR to RGB and display it in the first column
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width="always")
with col2:
    # Header for the compressed image, displaying the number of components and variance preserved
    st.header(f"Compressed Image ({pca_components} Components, {cumulative_variance[pca_components-1]:.2%} Variance Preserved)")
    # Display the compressed image in the second column
    st.image(re_image, use_column_width="always")

