import pathlib
import io
import base64

import cv2
import numpy as np
from PIL import Image

import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Generating a link to download a particular image file
def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format='JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Set title 
st.title('Image Restoration')

# Specify canvas parameters
uploaded_file = st.sidebar.file_uploader('Upload image to restore.', type=['png', 'jpg'])
image = None
res = None

if uploaded_file is not None:    
    # Convert image to openCV format
    raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # Loads image in a BGR channel order
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    stroke_width = st.sidebar.slider('Stroke width: ', 1, 25, 5)
    h, w = img.shape[:2]
    if w > 800:
        h_, w_ = int(h*800 / w), 800
    else:
        h_, w_ = h, w

    # Create a drawable canvas component
    canvas_result = st_canvas(
        fill_color='white',
        stroke_width=stroke_width,
        stroke_color='black',
        background_image=Image.open(uploaded_file).resize((h_, w_)),
        update_streamlit=True,
        height=h_,
        width=w_,
        drawing_mode='freedraw',
        key='canvas',
    )

    stroke = canvas_result.image_data

    if stroke is not None:
        if st.sidebar.checkbox('show mask'):
            st.image(stroke)

        mask = cv2.split(stroke)[3]
        mask = np.uint8(mask)
        mask = cv2.resize(mask, (w, h))
    
    st.sidebar.caption('Happy with the selection?')
    option = st.sidebar.selectbox('Mode', ['None', 'Telea', 'NS', 'Compare both'])

    if option == 'Telea':
        st.subheader('Result of Telea')
        res = cv2.inpaint(img, inpaintMask=mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)[:,:,::-1]
        st.image(res)
        result = Image.fromarray(res)
        st.sidebar.markdown(
            get_image_download_link(result, 'telea.png', 'Download output'),
            unsafe_allow_html=True)
    elif option == 'NS':
        st.subheader('Result of NS')
        res = cv2.inpaint(img, inpaintMask=mask, inpaintRadius=3, flags=cv2.INPAINT_NS)[:,:,::-1]
        st.image(res)
        result = Image.fromarray(res)
        st.sidebar.markdown(
            get_image_download_link(result, 'ns.png', 'Download output'),
            unsafe_allow_html=True)
    elif option == 'Compare both':
        col1, col2 = st.columns(2)
        telea = cv2.inpaint(img, inpaintMask=mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)[:,:,::-1]
        ns = cv2.inpaint(img, inpaintMask=mask, inpaintRadius=3, flags=cv2.INPAINT_NS)[:,:,::-1]
        with col1:
            st.subheader('Result of Telea')
            st.image(telea)
        with col2:
            st.subheader('Result of NS')
            st.image(ns)
        
        if telea is not None:
            result = Image.fromarray(telea)
            st.sidebar.markdown(
                get_image_download_link(result, 'telea.png', 'Download output'),
                unsafe_allow_html=True)

        if ns is not None:
            result = Image.fromarray(ns)
            st.sidebar.markdown(
                get_image_download_link(result, 'ns.png', 'Download output'),
                unsafe_allow_html=True)
    else:
        pass