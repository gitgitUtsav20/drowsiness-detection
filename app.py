import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import dlib
from imutils import face_utils
import numpy as np
import av
import os
import time
import base64

def get_audio_html():
    if not os.path.exists("alarm.wav"):
        return ""
    with open("alarm.wav", "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f"""
        <audio autoplay loop>
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
    """

audio_html = get_audio_html()

# Page configuration
st.set_page_config(page_title="Drowsiness Detection Web App", layout="centered")
st.title("Drowsiness Detection System 🛡️")
st.markdown("This web application uses your webcam to detect drowsiness in real-time.")

@st.cache_data
def get_ice_servers():
    """Use Twilio's TURN server to avoid firewall connection timeouts."""
    try:
        from twilio.rest import Client
        # Pull TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN from Streamlit's secrets
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        return token.ice_servers
    except Exception as e:
        # Fallback to standard STUN server if secrets are missing
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": get_ice_servers(),
        "iceTransportPolicy": "relay"
    }
)

@st.cache_resource
def load_models():
    # Verify that the model file exists
    model_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}! Please ensure the file is in the project directory.")
        
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)
    return detector, predictor

try:
    detector, predictor = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


def dist(a, b):
    x1, y1 = a
    x2, y2 = b
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


class DrowsinessDetector(VideoProcessorBase):
    def __init__(self):
        self.dlist = []
        self.thres = 6
        self.is_drowsy = False

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        rects = detector(gray, 0)
        
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # Draw landmarks
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            
            le_38 = shape[37]
            le_39 = shape[38]
            le_41 = shape[40]
            le_42 = shape[41]

            re_44 = shape[43]
            re_45 = shape[44]
            re_47 = shape[46]
            re_48 = shape[47]
            
            le_ear = dist(le_38, le_42) + dist(le_39, le_41)
            re_ear = dist(re_44, re_48) + dist(re_45, re_47)
            
            eye_aspect_ratio = (le_ear + re_ear) / 4
            
            self.dlist.append(eye_aspect_ratio < self.thres)
            if len(self.dlist) > 10:
                self.dlist.pop(0)
            
            # If drowsiness is detected
            drowsy = sum(self.dlist) >= 4
            self.is_drowsy = drowsy
            if drowsy:
                cv2.putText(image, "WARNING: DROWSY!", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

st.markdown("### Start your Webcam below:")
st.markdown("Click **START** to allow access to your webcam and begin drowsiness detection.")

webrtc_ctx = webrtc_streamer(
    key="drowsiness-detection",
    video_processor_factory=DrowsinessDetector,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

if webrtc_ctx.state.playing:
    status_placeholder = st.empty()
    audio_placeholder = st.empty()
    was_drowsy = False
    
    while True:
        if webrtc_ctx.video_processor:
            is_drowsy = webrtc_ctx.video_processor.is_drowsy
            if is_drowsy:
                status_placeholder.error("🚨 WARNING: YOU ARE DROWSY! 🚨")
                if not was_drowsy:
                    audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
                    was_drowsy = True
            else:
                status_placeholder.success("✅ Eyes Open - Status Normal")
                if was_drowsy:
                    audio_placeholder.empty()
                    was_drowsy = False
                    
        time.sleep(0.5)
