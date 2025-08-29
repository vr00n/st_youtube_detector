import streamlit as st
import cv2
import pafy
import torch
import numpy as np
import time
from PIL import Image

# --- Configuration ---
HOT_OBJECTS = ['person', 'cat', 'dog', 'laptop', 'cell phone', 'book', 'cup', 'pizza', 'car', 'bird']
HOTNESS_THRESHOLD = 0.65  # Minimum confidence to be considered "hot"
CAPTURE_COOLDOWN = 10  # Seconds to wait before capturing the same object class again

# --- Model Loading ---
# Cache the model loading to avoid reloading on every interaction.
@st.cache_resource
def load_yolo_model():
    """
    Loads the YOLOv5 model from PyTorch Hub.
    """
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv5 model: {e}")
        st.error("Please ensure you have an active internet connection to download the model.")
        return None

# --- Main App Logic ---
st.set_page_config(layout="wide", page_title="Hot or Not AI Detector")

st.title("🔥 Hot or Not AI Object Detector")
st.markdown("""
This application uses the YOLOv5 model to perform real-time object detection on a YouTube live stream.
Enter a YouTube video or stream URL below to begin analysis.
""")

# Load the model
model = load_yolo_model()

if model is None:
    st.stop()

# Initialize session state for storing captures and cooldowns
if 'captures' not in st.session_state:
    st.session_state.captures = []
if 'cooldowns' not in st.session_state:
    st.session_state.cooldowns = {}


# --- UI Elements ---
url = st.text_input("YouTube Stream URL:", "https://www.youtube.com/watch?v=jfKfPfyJRdk") # Example: A live NYC cam

if st.button("Start Analysis", type="primary"):
    if not url:
        st.warning("Please enter a URL.")
    else:
        try:
            # --- Video Stream Setup ---
            pafy_video = pafy.new(url)
            # Try to get the best quality stream, but fall back if needed
            play = pafy_video.getbest(preftype="mp4")
            if play is None:
                play = pafy_video.streams[-1] # Fallback to any available stream

            cap = cv2.VideoCapture(play.url)

            if not cap.isOpened():
                st.error("Error: Could not open video stream. Please check the URL and try another one.")
            else:
                # --- Create layout ---
                st.success(f"Successfully opened stream: **{pafy_video.title}**")
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
                        st.write("The video stream has ended.")
                        break

                    # Convert frame from BGR to RGB for YOLO model
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Perform detection
                    results = model(frame_rgb)
                    predictions = results.pandas().xyxy[0] # Get predictions as a pandas DataFrame

                    current_time = time.time()

                    # Process each detected object
                    for _, row in predictions.iterrows():
                        confidence = row['confidence']
                        class_name = row['name']
                        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

                        is_hot = class_name in HOT_OBJECTS and confidence >= HOTNESS_THRESHOLD

                        # --- Draw Bounding Boxes ---
                        color = (37, 99, 235) if is_hot else (245, 158, 11) # Blue for hot, Amber for not
                        label = f"{class_name.capitalize()} {confidence:.2f}"
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


                        # --- Capture Logic ---
                        last_capture_time = st.session_state.cooldowns.get(class_name, 0)
                        if is_hot and (current_time - last_capture_time > CAPTURE_COOLDOWN):
                            # Crop the detected object from the original frame
                            cropped_image = frame[ymin:ymax, xmin:xmax]
                            # Convert BGR (OpenCV) to RGB (PIL/Streamlit)
                            cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

                            # Create capture entry
                            capture_info = {
                                "image": Image.fromarray(cropped_image_rgb),
                                "class_name": class_name,
                                "hotness": confidence,
                                "timestamp": time.strftime("%H:%M:%S")
                            }

                            st.session_state.captures.insert(0, capture_info) # Add to the top of the list
                            st.session_state.cooldowns[class_name] = current_time # Update cooldown timer

                    # Display the processed frame
                    video_placeholder.image(frame, channels="BGR", use_column_width=True)

                    # Display the captures in the sidebar
                    with captures_placeholder:
                        if not st.session_state.captures:
                            st.info("No hot objects detected yet...")
                        else:
                            for i, capture in enumerate(st.session_state.captures):
                                st.image(capture['image'], caption=f"{capture['class_name'].capitalize()} ({capture['hotness']:.2f}) at {capture['timestamp']}", use_column_width=True)


                cap.release()

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("This could be due to an invalid URL, a private video, or a network issue.")
