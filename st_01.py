import streamlit as st
import pandas as pd
import plotly.express as px
import os
from PIL import Image
import base64

st.set_page_config(
    page_title = 'Drone Footage Object Detector',
    page_icon = '🚀',
    layout = 'wide'
)


st.markdown("""
    <style>
        .main {
            max-height: 100vh;
            overflow-y: scroll;
        }
    </style>
""", unsafe_allow_html=True)



st.markdown (  
    """
    <style>
    
    
    .block-container {
        padding-top: 0rem !important;
    }
    .title{
        font-size : 27px;
        font-family :  Georgia, serif;
        text-align : center;
        vertical-align: top;
        margin-top: 3px;
    }

    section[data-testid="stSidebar"] {
    background:linear-gradient(to bottom , #0F202B, #202D4A, #172626); ;
    padding: 20px;
    border-right: 2px solid #FFFFFF ;
    font-family: Verdana, sans-serif;
    font-size: 16px;
    width: 300px !important;  
    }
div[data-testid="stSidebarContent"] {
        width: 100% !important;
}

main{
      background: linear-gradient(to bottom right, #0F202B, #020229 );
      padding: 10px;
}
     
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown('<h1 class="title">Drone Footage Object Detection and Tracking</h1>', unsafe_allow_html=True)
st.sidebar.title("➤ Specifications : ")
for _ in range(15):
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.sidebar.markdown("""
    <p style='font-size:17px; font-weight:bold; font-family:"Segoe UI", sans-serif; color:white;'>
        ➤ Select Detection Model
    </p>
""", unsafe_allow_html=True)


selected_model = st.sidebar.radio(
    label = "",
    options=["SAM2", "YOLOv8 s-Worldv2"],
    index=1,  # default selected: YOLOv8s-Worldv2
    help="Choose the model to use for object detection."
)

st.markdown(
    '<p style="font-size:22px; font-family:\'Segoe UI\', sans-serif; font-weight:bold; color:#8cc8e6; margin-top:5px;">📸Upload a drone image</p>',
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader("", type=['jpg','jpeg','png','webp'])


if uploaded_file is not None:
    
     
    image = Image.open(uploaded_file)

    st.markdown("### 🖼️ Uploaded Image Preview")

    image_bytes = uploaded_file.getvalue()
    encoded = base64.b64encode(image_bytes).decode()

    mime_type = uploaded_file.type  
    
    st.markdown(
        """
        <style>
        .image-box {
            border: 2px solid #888;
            padding: 10px;
            width: 550px;
            height: 450px;
            overflow: hidden;
            border-radius: 10px;
            background-color: #f5f5f5;
        }
        .image-box img {
            width: 100%;
            height: auto;
            object-fit: contain;
        }
        </style>
        """, unsafe_allow_html=True
    )

    
    st.markdown(f"""
        <div class="image-box">
            <img src="data:{mime_type};base64,{encoded}" />
        </div>
    """, unsafe_allow_html=True)