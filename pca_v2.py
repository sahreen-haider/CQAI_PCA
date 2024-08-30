import numpy as np  
import cv2  
import streamlit as st 
from sklearn.decomposition import PCA  
import os, random  
from utils import get_image_list, load_image, apply_pca

st.set_page_config(layout="wide")

# Check if 'random_file' is in session state, initialize it if not
if 'random_file' not in st.session_state:
    st.session_state.random_file = "cat2.jpg"

st.title("Usage of Principal Component Analysis (PCA) in Image Compression")

# Sidebar widget for file upload with a label for accessibility
uploaded_file = st.sidebar.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg', 'tiff', 'bmp'], label_visibility="collapsed")

if uploaded_file is not None:
    st.sidebar.info(f"File uploaded: {uploaded_file.name}")
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Decode the image in BGR mode
else:
    if st.sidebar.button("Try New Example Image!"):
        random_file = random.choice(get_image_list())
        st.session_state.random_file = random_file

    st.sidebar.info(f"Example file: {st.session_state.random_file}")
    image = load_image(f"Image/{st.session_state.random_file}")  # Load the selected example image in BGR mode

# Split the image into its blue, green, and red channels (BGR format)
blue, green, red = cv2.split(image)
st.sidebar.write(blue.shape)

# Perform PCA on the blue channel to calculate explained variance
pca_temp = PCA().fit(blue)
explained_variance_ratio = pca_temp.explained_variance_ratio_

# Calculate cumulative variance for the PCA components
cumulative_variance = np.cumsum(explained_variance_ratio)

# Function to generate a slider label with the number of components and variance information
def slider_label(components):
    return f"Number of PCA Components: {components}, Variance Preserved: {cumulative_variance[components-1]:.2%}"

# Create a slider for selecting the number of PCA components, with the variance information displayed
pca_components = st.slider(slider_label(20), 1, blue.shape[0], 20, format="%d")
st.sidebar.write(f"Variance preserved: {cumulative_variance[pca_components-1]:.2%}")

# Apply PCA to each channel using the custom utility function
redI = apply_pca(red, pca_components)
greenI = apply_pca(green, pca_components)
blueI = apply_pca(blue, pca_components)

# Reconstruct the image by stacking the processed channels (still in BGR order) and converting it to an unsigned 8-bit integer format
re_image_bgr = (np.dstack((blueI, greenI, redI))).astype(np.uint8)

# Convert the reconstructed image from BGR to RGB for correct color representation
re_image_rgb = cv2.cvtColor(re_image_bgr, cv2.COLOR_BGR2RGB)

# Create two columns to display the original and compressed images side by side
col1, col2 = st.columns(2)
with col1:
    st.header("Original Image")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width="always")  # Convert the original image to RGB for correct display
with col2:
    st.header(f"Compressed Image ({pca_components} Components, {cumulative_variance[pca_components-1]:.2%} Variance Preserved)")
    st.image(re_image_rgb, use_column_width="always")  # Display the compressed image in RGB format
