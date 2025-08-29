import streamlit as st
import cv2
import torch
import numpy as np
import time
from PIL import Image
import yt_dlp

# --- Configuration ---
HOT_OBJECTS = ['person', 'cat', 'dog', 'laptop', 'cell phone', 'book', 'cup', 'pizza', 'car', 'bird']
HOTNESS_THRESHOLD = 0.65  # Minimum confidence to be considered "hot"
CAPTURE_COOLDOWN = 10  # Seconds to wait before capturing the same object class again

# --- Model Loading ---
@st.cache_resource
def load_yolo_model():
    """
    Loads the YOLOv5 model from PyTorch Hub.
    Using a cached resource to prevent reloading on every script rerun.
    """
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv5 model: {e}")
        st.error("Please ensure you have an active internet connection and all dependencies from requirements.txt are installed.")
        return None

# --- Stream URL Fetching ---
def get_stream_url(youtube_url):
    """
    Uses yt-dlp to extract the direct stream URL from a YouTube URL.
    Specifies a format that OpenCV can reliably handle.
    """
    ydl_opts = {
        # Request the best quality format that is a direct MP4 file, which is more compatible with OpenCV
        'format': 'best[ext=mp4]/best',
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            return info_dict.get('url', None), info_dict.get('title', 'Untitled Stream')
    except Exception as e:
        st.error(f"yt-dlp error: Could not fetch stream info. The stream may not be available in a compatible format. Error: {e}")
        return None, None

# --- Main App Logic ---
st.set_page_config(layout="wide", page_title="Hot or Not AI Detector")

st.title("🔥 Hot or Not AI Object Detector")
st.markdown("""
This application uses the YOLOv5 model to perform real-time object detection on a YouTube live stream.
It has been updated to use `yt-dlp` with specific formats for improved reliability with OpenCV.
""")

model = load_yolo_model()
if model is None:
    st.stop()

# Initialize session state for storing captures and cooldown timers
if 'captures' not in st.session_state:
    st.session_state.captures = []
if 'cooldowns' not in st.session_state:
    st.session_state.cooldowns = {}

# --- UI Elements ---
url = st.text_input("YouTube Stream URL:", "https://www.youtube.com/watch?v=jfKfPfyJRdk") # Example: Live NYC cam

if st.button("Start Analysis", type="primary"):
    if not url:
        st.warning("Please enter a URL.")
    else:
        st.session_state.captures = [] # Clear previous captures on new run
        st.session_state.cooldowns = {}
        
        with st.spinner('Attempting to open stream...'):
            stream_url, stream_title = get_stream_url(url)

        if stream_url:
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                st.error("Error: Could not open video stream. Please check the URL and try another one.")
            else:
                st.success(f"Successfully opened stream: **{stream_title}**")
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.header("Live Feed")
                    video_placeholder = st.empty()

                with col2:
                    st.header("📸 Hot Catches")
                    captures_placeholder = st.container()

                # --- Detection Loop ---
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.write("The video stream has ended or was interrupted.")
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = model(frame_rgb)
                    predictions = results.pandas().xyxy[0]
                    current_time = time.time()

                    for _, row in predictions.iterrows():
                        confidence = row['confidence']
                        class_name = row['name']
                        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

                        is_hot = class_name in HOT_OBJECTS and confidence >= HOTNESS_THRESHOLD
                        color = (37, 99, 235) if is_hot else (245, 158, 11) # Blue-600 for hot, Amber-500 for not
                        label = f"{class_name.capitalize()} {confidence:.2f}"
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        last_capture_time = st.session_state.cooldowns.get(class_name, 0)
                        if is_hot and (current_time - last_capture_time > CAPTURE_COOLDOWN):
                            cropped_image = frame[ymin:ymax, xmin:xmax]
                            cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

                            capture_info = {
                                "image": Image.fromarray(cropped_image_rgb),
                                "class_name": class_name,
                                "hotness": confidence,
                                "timestamp": time.strftime("%H:%M:%S")
                            }
                            st.session_state.captures.insert(0, capture_info)
                            st.session_state.cooldowns[class_name] = current_time

                    video_placeholder.image(frame, channels="BGR", use_column_width=True)

                    with captures_placeholder:
                        # This part can be slow if redrawn constantly. Let's optimize.
                        # We can rebuild the capture list only when a new one is added.
                        # For now, keeping it simple.
                        if not st.session_state.captures:
                            st.info("No hot objects detected yet...")
                        else:
                            # To prevent the list from growing indefinitely, let's cap it
                            for capture in st.session_state.captures[:20]: # Show latest 20
                                st.image(capture['image'], caption=f"{capture['class_name'].capitalize()} ({capture['hotness']:.2f}) at {capture['timestamp']}", use_column_width=True)
                
                cap.release()

