#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('vgg2_model.h5')

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess the input image
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    img = cv2.resize(img, (224, 224))  # Resize to match input shape of the model
    img = img / 255.0  # Normalize pixel values
    return img

# Function to predict BMI from an image
def predict_bmi(image):
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)[0][0]
    return prediction

# Create a Streamlit app
st.title("BMI Estimation")

# Open the webcam
cap = cv2.VideoCapture(0)

# Create a placeholder for the image
image_placeholder = st.empty()

# Process and display frames in the Streamlit app
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face region from the frame
            face_img = frame[y:y + h, x:x + w]

            # Make prediction
            bmi = predict_bmi(face_img)

            # Display the bounding box around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the BMI on the frame
            cv2.putText(frame, f'BMI: {bmi:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame in Streamlit
        image_placeholder.image(frame, channels="BGR")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()

