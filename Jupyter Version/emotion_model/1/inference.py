import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import base64
import json
import pandas as pd

# Load your songs dataframe here, for example:
 Music_Player = pd.read_csv('songs1.csv')
#Music_Player = pd.DataFrame(...)  # Replace with your actual dataframe

# Load the face cascade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def load_and_prep_image(image_data, img_shape=224):
    # The image is a base64 encoded string
    image_data = base64.b64decode(image_data)
    img_array = np.fromstring(image_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    GrayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(GrayImg, 1.1, 4)
    for x, y, w, h in faces:
        roi_color = img[y:y+h, x:x+w]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        img = roi_color

    RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    RGBImg = cv2.resize(RGBImg, (img_shape, img_shape))
    RGBImg = RGBImg / 255.0

    return np.expand_dims(RGBImg, axis=0)

def Recommend_Songs(pred_class):
    if pred_class == 'Disgust':
        Play = Music_Player[Music_Player['Mood'] == 'Sad']
    elif pred_class in ['Happy', 'Sad']:
        Play = Music_Player[Music_Player['Mood'] == 'Happy']
    elif pred_class in ['Fear', 'Angry']:
        Play = Music_Player[Music_Player['Mood'] == 'Calm']
    elif pred_class in ['Surprise', 'Neutral']:
        Play = Music_Player[Music_Player['Mood'] == 'Energetic']
    else:
        # Provide a default behavior if an unknown class is predicted
        Play = Music_Player[Music_Player['Mood'] == 'Neutral']

    Play = Play.sample(n=5).reset_index(drop=True)
    # Convert the pandas dataframe to a JSON format
    recommendations_json = Play.to_json(orient='records')
    return json.loads(recommendations_json)

# Rest of the SageMaker inference code (model_fn, input_fn, predict_fn, output_fn)
# ...
def input_fn(request_body, content_type='application/x-image'):
    if content_type == 'application/x-image':
        img_data = base64.b64decode(request_body)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))  # Resize as per model's expected input
        img = img / 255.0  # Normalize if your model expects data in this range
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    else:
        raise ValueError(f'Unsupported content type: {content_type}')


# Output processing function
def output_fn(prediction_output, accept='application/json'):
    if accept == 'application/json':
        pred_class = np.argmax(prediction_output)
        # Here, you would use your Recommend_Songs function to get recommendations based on the predicted class
        recommendations = Recommend_Songs(pred_class)
        # You would create a JSON response containing the predictions and recommendations
        return json.dumps(recommendations)
    else:
        raise ValueError(f'Unsupported accept type: {accept}')

# The predict function
def predict_fn(input_data, model):
    # Make a prediction
    prediction = model.predict(input_data)
    return prediction
