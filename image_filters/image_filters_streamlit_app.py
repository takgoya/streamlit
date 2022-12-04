import io
import base64

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from image_filters import *

# Generating a link to download a particular image file
def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format='JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

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
                          'Sepia / Vintage',
                          'Vignette Effect',
                          'Edge Detection',
                          'Brightness',
                          'Contrast',
                          'Embossed Edges',
                          'Outline Filter',
                          'Pencil Sketch',
                          'Stylization Filter') )

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
    elif option == 'Vignette Effect':
        level = st.slider('level', min_value=1, max_value=5, value=2)
        img_output = vignette_filter(img, level)
    elif option == 'Edge Detection':
        lower_thresh = st.slider('lower_thresh', min_value=10, max_value=150, step=5, value=50 )
        upper_thresh = st.slider('thrupper_threshesh2', min_value=lower_thresh, max_value=255, step=5, value=200)
        img_output = edge_detection_filter(img, lower_thresh, upper_thresh)
        output_color = 'GRAY'
    elif option == 'Brightness':
        brightness = st.slider('brightness', min_value=-50, max_value=50, step=10, value=0)
        img_output = brightness_filter(img, brightness)
    elif option == 'Contrast':
        contrast = st.slider('contrast', min_value=0.2, max_value=2.0, step=0.2, value=1.0)
        img_output = contrast_filter(img, contrast)
    elif option == 'Embossed Edges':
        img_output = embossed_edges_filter(img)
    elif option == 'Outline Filter':
        k = st.slider('k', min_value=9, max_value=12, step=1, value=9)
        img_output = outline_filter(img, k)
    elif option == 'Pencil Sketch':
        kernel = st.slider('kernel', min_value=1, max_value=9, step=2, value=5)
        img_output = pencil_sketch_filter(img)
        output_color = 'GRAY'
    elif option == 'Stylization Filter':
        img_output = stylization_filter(img)


    with output_col:
        if output_display == True:
            st.header(option)
            st.image(img_output, channels=output_color, use_column_width=True)

            # fromarray convert cv2 image into PIL format for saving it using download link.
            if output_color == 'BGR':
                result = Image.fromarray(img_output[:,:,::-1])
            else:
                result = Image.fromarray(img_output)
            # Display link.
            st.markdown(get_image_download_link(result,'output.png','Download '+'Output'),
                        unsafe_allow_html=True)

