import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Emotion Detection System", layout="centered")
st.title("Real-Time Emotion Detection System")
st.write("This application uses a Deep Learning CNN model to monitor and track emotional states live.")

@st.cache_resource
def load_emotion_model():
    return load_model("Emotion_detection_model_new.h5")

try:
    model = load_emotion_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

emotion_labels = {
    0: "Angry",
    1: "Happy",
    2: "Neutral",
    3: "Sad",
    4: "Surprised"
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- FIX: Use Streamlit's native browser webcam input component ---
img_file_buffer = st.camera_input("Take a photo to analyze your current stress/emotion level")

if img_file_buffer is not None:
    # Convert the file buffer into a format OpenCV understands
    bytes_data = img_file_buffer.getvalue()
    cv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Preprocessing pipeline matching your exact model logic
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        st.warning("No face detected in the frame. Please look directly into the camera in a well-lit area.")
    
    for (x, y, w, h) in faces:
        # Crop, resize, normalize, and reshape
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = face.reshape(1, 48, 48, 1)
        
        # Run Inference
        prediction = model.predict(face, verbose=0)
        emotion = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        # Draw dynamic overlay graphics on the frame
        cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(
            cv_image, 
            f"{emotion} ({confidence:.1f}%)", 
            (x, y-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            (0, 255, 0), 
            2
        )
        
    # Convert BGR back to RGB for crisp browser rendering
    final_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    # Render the calculated output frame below the camera feed widget
    st.image(final_image, caption="Processed Analysis State", use_column_width=True)
