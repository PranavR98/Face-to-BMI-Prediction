#!/usr/bin/env python
# coding: utf-8

# In[3]:

import numpy as np
import streamlit as st
import cv2
import tensorflow as tf

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the BMI prediction model
model = tf.keras.models.load_model('vgg2_model.h5')

# Define a function to preprocess the input image
def preprocess_image(image):
    # Resize the image to the required input size of your model
    resized_image = cv2.resize(image, (224, 224))
    # Normalize the image pixel values
    normalized_image = resized_image / 255.0
    # Reshape the image to add a batch dimension
    final_image = normalized_image.reshape((1, 224, 224, 3))
    return final_image

# Define the Streamlit app
def app():
    # Add a title to the app
    st.title("BMI Prediction App")
    
    # Add a file uploader to allow the user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # If the user has uploaded an image
    if uploaded_file is not None:
        # Read the image from the uploaded file
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Convert the image to grayscale for face detection
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # For each detected face
        for (x, y, w, h) in faces:
            # Extract the face region from the image
            face_image = image[y:y+h, x:x+w]
            
            # Preprocess the face image
            preprocessed_image = preprocess_image(face_image)
            
            # Make a prediction using the loaded model
            prediction = model.predict(preprocessed_image)
            
            # Calculate the predicted BMI
            bmi = prediction[0][0]
            
            # Draw the bounding box around the face in the original image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add the predicted BMI inside the bounding box
            cv2.putText(image, f"BMI: {bmi:.2f}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display the modified image with the bounding boxes and predicted BMIs
        st.image(image, caption='Uploaded Image', use_column_width=True, channels='BGR')
        
# Run the app
app()







# In[ ]:




