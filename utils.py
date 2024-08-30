import streamlit as st
import cv2
import os
from sklearn.decomposition import PCA, KernelPCA

# Cache the directory listing to avoid repeated I/O operations
@st.cache_resource
def get_image_list(directory="Image"):
    return os.listdir(directory)

# Cache the image reading to avoid repeated decoding
@st.cache_resource
def load_image(file_path):
    return cv2.imread(file_path)

# Function to perform PCA and inverse transform
def apply_pca(channel, components):
    pca = PCA(components)
    transformed = pca.fit_transform(channel)
    return pca.inverse_transform(transformed)

def apply_kernel_pca(channel, components, kernel="rbf"):
    kpca = KernelPCA(n_components=components, kernel=kernel, fit_inverse_transform=True)
    transformed = kpca.fit_transform(channel)
    return kpca.inverse_transform(transformed)

