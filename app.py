from pathlib import Path
from PIL import Image
import streamlit as st
import os
import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam_cloud, infer_uploaded_webcam_local

IS_LOCAL = os.environ.get('IS_LOCAL', 'false').lower() == 'true'

# setting page layout
st.set_page_config(
    page_title="Deep Sea Vision",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# main page heading
st.title("Underwater Object Detection")

# Add logo to sidebar
st.sidebar.image('logo.png', width=300)  # Adjust width as needed

# sidebar
st.sidebar.header("DL Model Config")

# model options
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection"]
)

model_type = None
if task_type == "Detection":
    model_type = st.sidebar.selectbox(
        "Select Model",
        config.DETECTION_MODEL_LIST
    )
else:
    st.error("Currently only 'Detection' function is implemented")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100

model_path = ""
if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
else:
    st.error("Please Select Model in Sidebar")

# load pretrained DL model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    config.SOURCES_LIST
)

source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # Image
    infer_uploaded_image(confidence, model)
elif source_selectbox == config.SOURCES_LIST[1]: # Video
    infer_uploaded_video(confidence, model)
elif source_selectbox == config.SOURCES_LIST[2]:  # Webcam
    if IS_LOCAL:
        infer_uploaded_webcam_local(confidence, model)
    else:
        infer_uploaded_webcam_cloud(confidence, model)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")