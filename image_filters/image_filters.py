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

