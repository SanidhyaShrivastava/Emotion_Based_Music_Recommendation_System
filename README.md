# Emotion-Based Music Recommendation System

This web application uses emotion analysis from text and images to recommend music. It leverages AWS Lambda functions for sentiment analysis from text and emotion analysis from images, integrating with the Amazon Rekognition service for image-based emotion detection. Additionally, the application provides a feature to capture emotions from a live camera feed using the user's webcam.

## Features

- **Sentiment Analysis**: Users can input text, and the application will analyze the sentiment to recommend music based on the detected sentiment.
- **Image-based Emotion Analysis**: Users can upload an image, and the application will detect the emotion from the face in the image to recommend music.
- **Live Camera Emotion Detection**: Users can start their webcam, capture an image, and the application will analyze the emotion from the captured image to recommend music.

## How It Works

1. **Sentiment Analysis**: 
   - The user inputs text into a textarea.
   - The text is sent to an AWS Lambda function for sentiment analysis.
   - Based on the detected sentiment, the application fetches a music recommendation and displays it.

2. **Image-based Emotion Analysis**:
   - The user uploads an image.
   - The image is sent to an AWS Lambda function, where Amazon Rekognition is used to analyze the emotion from any faces in the image.
   - Based on the detected emotion, the application fetches a music recommendation and displays it.

3. **Live Camera Emotion Detection**:
   - The user can start their webcam using the provided button.
   - Once the camera is on, the user can capture an image.
   - The captured image is sent to an AWS Lambda function for emotion analysis using Amazon Rekognition.
   - Based on the detected emotion, the application fetches a music recommendation and displays it.
