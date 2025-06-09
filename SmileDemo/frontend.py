# Frontend.py

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import cv2
import numpy as np
import requests
import base64
from PIL import Image
import io
import face_recognition

# --- Page Configuration ---
st.set_page_config(
    page_title="Smile Counter",
    page_icon="😊",
    layout="wide",
)

# --- Backend API URL ---
API_URL = "http://127.0.0.1:8000/detect"

# --- Session State Initialization ---
if 'smile_count' not in st.session_state:
    st.session_state.smile_count = 0
if 'known_face_encodings' not in st.session_state:
    st.session_state.known_face_encodings = []
if 'smiled_faces' not in st.session_state:
    st.session_state.smiled_faces = set()

# --- UI Layout ---
st.title("😊 Smile Counter")
st.markdown("This application uses AI to detect smiles from your webcam. Your smile will only be counted once.")

# Custom CSS for layout
st.markdown("""
<style>
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .counter {
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
header = st.container()
with header:
    st.markdown('<div class="header">', unsafe_allow_html=True)
    settings_col, counter_col = st.columns([1, 1])
    
    with settings_col:
        with st.expander("Settings"):
            st.slider("Face Match Tolerance", 0.1, 1.0, 0.6, 0.05, key="tolerance")
            if st.button("Reset Smile Count"):
                st.session_state.smile_count = 0
                st.session_state.known_face_encodings = []
                st.session_state.smiled_faces = set()
                st.experimental_rerun()

    with counter_col:
        st.markdown('<div style="text-align: right;">', unsafe_allow_html=True)
        st.write("Live Smile Count:")
        smile_counter_placeholder = st.empty()
        smile_counter_placeholder.markdown(f'<p class="counter">{st.session_state.smile_count}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)


# --- WebRTC Video Transformer ---
class SmileDetector(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Process every 5th frame to save resources
        self.frame_count += 1
        if self.frame_count % 5 != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Convert image to bytes
        _, img_encoded = cv2.imencode(".jpg", img)
        files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}

        try:
            response = requests.post(API_URL, files=files, timeout=0.5)
            response.raise_for_status()
            data = response.json()

            for face in data['faces']:
                box = face['box']
                encoding = np.array(face['encoding'])
                is_smiling = face['is_smiling']
                
                # Draw bounding box
                color = (0, 255, 0) if is_smiling else (0, 0, 255)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                
                # Check if this face is already known
                if len(st.session_state.known_face_encodings) > 0:
                    matches = face_recognition.compare_faces(st.session_state.known_face_encodings, encoding, tolerance=st.session_state.tolerance)
                    face_distances = face_recognition.face_distance(st.session_state.known_face_encodings, encoding)
                    
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        # It's a known face
                        if is_smiling and best_match_index not in st.session_state.smiled_faces:
                            st.session_state.smile_count += 1
                            st.session_state.smiled_faces.add(best_match_index)
                            smile_counter_placeholder.markdown(f'<p class="counter">{st.session_state.smile_count}</p>', unsafe_allow_html=True)
                    else:
                        # It's a new face
                        st.session_state.known_face_encodings.append(encoding)
                        if is_smiling:
                            st.session_state.smile_count += 1
                            st.session_state.smiled_faces.add(len(st.session_state.known_face_encodings) - 1)
                            smile_counter_placeholder.markdown(f'<p class="counter">{st.session_state.smile_count}</p>', unsafe_allow_html=True)

                else:
                    # This is the first face detected
                    st.session_state.known_face_encodings.append(encoding)
                    if is_smiling:
                        st.session_state.smile_count += 1
                        st.session_state.smiled_faces.add(0)
                        smile_counter_placeholder.markdown(f'<p class="counter">{st.session_state.smile_count}</p>', unsafe_allow_html=True)

        except (requests.RequestException, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            # Handle cases where backend is not available
            print(f"Could not connect to backend: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- Main Application ---
webrtc_ctx = webrtc_streamer(
    key="smile-detector",
    video_transformer_factory=SmileDetector,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if not webrtc_ctx.state.playing:
    st.info("Click 'START' to begin smile detection.")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and FastAPI.")
