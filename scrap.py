import numpy as np  
import cv2  
import streamlit as st
from sklearn.decomposition import PCA  
import os, random  
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image


img = Image.open("Image\cat1.jpg")

img = np.array(img)

blue, green, red = cv2.split(img)

plt.hist(blue.flatten(), bins=256, range = (0, 1), color='gray')
plt.show()

blue, green, red = blue/255.0, green/255.0, red/255.0

# sns.lineplot(blue)
# plt.show()

print(type(blue))