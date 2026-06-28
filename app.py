import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import time

# ==========================================
# 1. STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Real-Time Student Emotion Tracker",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS to force the camera output to stay a fixed size and not stretch fullscreen
st.markdown(
    """
    <style>
    .fixed-camera img {
        width: 640px !important;
        height: 480px !important;
        object-fit: cover;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 2. SIDEBAR - ABOUT THE PROJECT & CANDIDATE
# ==========================================
st.sidebar.markdown("## 🎓 Evaluation Profile")
st.sidebar.info(
    """
    **Institution:** Prananath Autonomous College, Khordha  
    Department of Computer Science  
    
    **Programme:** B.Sc. Computer Science (H), 4th Semester  
    *(2024 Admission Batch)*
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 👤 Candidate Details")
st.sidebar.markdown(
    """
    * **Name:** Nikhilesh Palei & Chinmayee Routray
    * **Project Topic:** Student Stress Detection & Coping Mechanism Awareness using Emotion Detection System  
    * **Training Dataset:** 16,082 Image FER-2013 Filtered Dataset
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🛠️ System Architecture")
st.sidebar.markdown(
    """
    - **Backend Engine:** Native OpenCV + TensorFlow Keras  
    - **Inference Mode:** Local Hardware Optimization (Zero Internet Lag)    
    """
)

# ==========================================
# 3. MAIN INTERFACE FRAMEWORK
# ==========================================
st.title("Real-Time Student Stress & Emotion Detection System")
st.subheader("Local Inference Engine — OpenCV Integration Window")
st.write("Developed By :  Nikhilesh Palei  &  Chinmayee Routray")
st.markdown("---")

# Safe loading of model assets
@st.cache_resource
def load_deep_learning_assets():
    model_path = "Emotion_detection_model_new.h5"
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file '{model_path}' not found in current directory! Please check your path.")
        return None, None
        
    model = tf.keras.models.load_model(model_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return model, face_cascade

model, face_cascade = load_deep_learning_assets()

# FIXED: Standardized to match the variable inside the camera loop exactly
emotion_labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

# ==========================================
# 4. CAMERA RUNTIME OPERATION
# ==========================================
if model is not None:
    st.markdown("### 🎥 Live Video Processing Feed")
    
    run_cam = st.checkbox("Toggle Live Webcam Stream", value=False)
    
    # FIXED: Placed inside a designated HTML div snippet to lock its dimensions down
    st.markdown('<div class="fixed-camera">', unsafe_allow_html=True)
    frame_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
    if run_cam:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Error: Could not access the webcam device channel.")
            run_cam = False

        while run_cam:
            ret, frame = cap.read()
            if not ret:
                st.warning("Stopped receiving frames from camera channel.")
                break
                
            frame = cv2.flip(frame, 1)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (48, 48))
                face_roi = face_roi / 255.0
                face_roi = face_roi.reshape(1, 48, 48, 1)
                
                # Predict
                predictions = model.predict(face_roi, verbose=0)
                max_index = np.argmax(predictions)
                
                # FIXED: Fixed variable name casing typo to resolve NameError completely
                emotion = emotion_labels[max_index]
                confidence = np.max(predictions) * 100
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame, 
                    f"{emotion} ({confidence:.1f}%)", 
                    (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (0, 255, 0), 
                    2
                )
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # FIXED: Removed fullscreen stretching flag so custom style layout rules take absolute priority
            frame_placeholder.image(frame_rgb, channels="RGB")
            
            time.sleep(0.01)
            
        cap.release()
        frame_placeholder.empty()
        st.write("🔴 Live feed standby. Check box to activate camera.")
    else:
        st.write("🔴 Live feed standby. Check box to activate camera.")