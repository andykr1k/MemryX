import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import cv2
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Smile AI", layout="wide")

# --- Init session state ---
for key, default in {
    'smile_count': 0,
    'known_face_encodings': [],
    'smiled_faces': set(),
    'tolerance': 0.6,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

API_URL = "http://127.0.0.1:8000/detect"

# --- Custom Header ---
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container { padding: 0; margin: 0; }
.app-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 1rem 2rem; background-color: #111; color: white; position: fixed;
    top: 0; width: 100%; z-index: 1000;
}
.logo-title { display: flex; align-items: center; gap: 1rem; }
.logo-title img { height: 40px; }
.smile-count { font-size: 1.5rem; font-weight: bold; }
.spacer { height: 80px; }
</style>
""", unsafe_allow_html=True)

header_placeholder = st.empty()
def update_header():
    header_placeholder.markdown(f"""
    <div class="app-header">
        <div class="spacer"></div>
        <div class="logo-title">
            <img src="https://memryx.com/wp-content/uploads/2021/05/MemryX-logo.svg">
        </div>
        <div class="smile-count">😊 {st.session_state.smile_count}</div>
    </div>
    <div class="spacer"></div>
    """, unsafe_allow_html=True)

update_header()

# --- Sidebar Settings ---
with st.sidebar:
    st.subheader("Settings")
    st.slider("Face Match Threshold", 0.1, 1.0, st.session_state["tolerance"], 0.05, key="tolerance")
    if st.button("Reset Smile Count"):
        st.session_state.smile_count = 0
        st.session_state.known_face_encodings = []
        st.session_state.smiled_faces = set()
        update_header()

# --- Smile Detector Transformer ---
class SmileDetector(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        if self.frame_count % 5 != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        _, img_encoded = cv2.imencode(".jpg", img)
        files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}

        try:
            r = requests.post(API_URL, files=files, timeout=1)
            r.raise_for_status()
            faces = r.json()['faces']

            for face in faces:
                box = face['box']
                encoding = np.array(face['encoding']).reshape(1, -1)
                is_smiling = face['is_smiling']
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),
                              (0, 255, 0) if is_smiling else (0, 0, 255), 2)

                encodings = st.session_state.known_face_encodings
                if encodings:
                    sims = cosine_similarity(np.array(encodings), encoding).flatten()
                    best_idx = np.argmax(sims)
                    if sims[best_idx] > st.session_state.tolerance:
                        if is_smiling and best_idx not in st.session_state.smiled_faces:
                            st.session_state.smile_count += 1
                            st.session_state.smiled_faces.add(best_idx)
                    else:
                        encodings.append(encoding.flatten())
                        if is_smiling:
                            idx = len(encodings) - 1
                            st.session_state.smile_count += 1
                            st.session_state.smiled_faces.add(idx)
                else:
                    encodings.append(encoding.flatten())
                    if is_smiling:
                        st.session_state.smile_count += 1
                        st.session_state.smiled_faces.add(0)

                update_header()
        except Exception as e:
            print(f"Error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Start Stream ---
webrtc_streamer(
    key="smile-detector",
    video_transformer_factory=SmileDetector,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
