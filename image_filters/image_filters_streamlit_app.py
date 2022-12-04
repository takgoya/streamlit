import os

import cv2
import numpy as np
import streamlit as st

from image_filters import black_white_filter, sepia_filter


# Set a title
st.title('Image Filters')

# Upload an image
img_uploaded = st.file_uploader('Choose an image file:', type=['png', 'jpg'])

if img_uploaded is not None:
    # Convert image to openCV format
    raw_bytes = np.asarray(bytearray(img_uploaded.read()), dtype=np.uint8)
    # Loads image in a BGR channel order
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    # Create two columns for displaying images
    input_col, output_col = st.columns(2)

    # Display uploaded image
    with input_col:
        st.header('Original')
        st.image(img_uploaded, channels='BGR', use_column_width=True)

    # Display a selection box for choosing the filter to apply
    option = st.selectbox('Select a filter to apply:', 
                          ('None', 
                          'Black & White',
                          'Sepia / Vintage'))

    # Flag for showing the image
    output_display = True
    # Colorspace of output image
    output_color = 'BGR'

    # Generate filtered image based on the selected option
    if option == 'None':
        # Do not show image
        output_display = False
    elif option == 'Black & White':
        img_output = black_white_filter(img)
        output_color = 'GRAY'
    elif option == 'Sepia / Vintage':
        img_output = sepia_filter(img)

    with output_col:
        if output_display == True:
            st.header(option)
            st.image(img_output, channels=output_color, use_column_width=True)
