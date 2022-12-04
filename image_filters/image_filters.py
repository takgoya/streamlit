import os

import cv2
import numpy as np
import streamlit as st

@st.cache
def black_white_filter(img):
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_bw

@st.cache
def sepia_filter(img):
    img_sepia = img.copy()
    # Convert to RGB as sepia matrix below is for RGB.
    img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_BGR2RGB)
    img_sepia = np.array(img_sepia, dtype=np.float64)
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.393, 0.769, 0.189],
                                                    [0.349, 0.686, 0.168],
                                                    [0.272, 0.534, 0.131]]))
    # Clip values to range [0, 255]
    img_sepia = np.clip(img_sepia, 0, 255)
    img_sepia = np.array(img_sepia, dtype=np.uint8)
    img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_RGB2BGR)

    return img_sepia

@st.cache
def vignette_filter(img, level=2):
    height, width = img.shape[:2]

    # Generate vignette mask using Gaussian kernels
    X_resultant_kernel = cv2.getGaussianKernel(width, width/level)
    Y_resultant_kernel = cv2.getGaussianKernel(height, height/level)

    # Generate resultant_kernel matrix
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()

    img_vignette = img.copy()

    # Applying the mask to each channel in the input image
    for i in range(3):
        img_vignette[:,:,i] = img_vignette[:,:,i] * mask 

    return img_vignette

@st.cache
def edge_detection_filter(img, lower_thresh, upper_thresh):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img_gray, (5,5), 0, 0)
    img_thresh = cv2.Canny(img_blurred, lower_thresh, upper_thresh)
    
    return img_thresh

@st.cache
def brightness_filter(img, brigthness):
    img_bright = cv2.convertScaleAbs(img, beta=brigthness)
    return img_bright

@st.cache
def contrast_filter(img, contrast):
    img_contrast = cv2.convertScaleAbs(img, alpha=contrast)
    return img_contrast

@st.cache
def embossed_edges_filter(img):
    kernel = np.array([[0, -3, -3],
                       [3, 0, -3],
                       [3, 3, 0]])

    img_embossed = cv2.filter2D(img, -1, kernel=kernel)

    return img_embossed

@st.cache
def outline_filter(img, k):
    k = max(k, 9)
    kernel = np.array([[-1, -1, -1],
                       [-1, k, -1],
                       [-1, -1, -1]])

    img_outline = cv2.filter2D(img, ddepth=-1, kernel=kernel)

    return img_outline

@st.cache
def pencil_sketch_filter(img, kernel=5):
    img_blur = cv2.GaussianBlur(img, (kernel, kernel), 0, 0)
    img_sketch, _ = cv2.pencilSketch(img_blur)
    return img_sketch

@st.cache
def stylization_filter(img):
    img_blur = cv2.GaussianBlur(img, (5, 5), 0, 0)
    img_style = cv2.stylization(img_blur, sigma_s=40, sigma_r=0.1)
    return img_style
