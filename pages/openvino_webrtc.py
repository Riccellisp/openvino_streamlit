import streamlit as st 
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
from openvino.runtime import Core
import operator
from Emotions import openvino_emotion



def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    # any operation 
    #flipped = img[::-1,:,:]
    pred_img = openvino_emotion(img)
    st.write(pred_img)

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="example", 
                video_frame_callback=video_frame_callback,
                media_stream_constraints={"video":True,"audio":False})