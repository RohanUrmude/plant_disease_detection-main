import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from werkzeug.utils import secure_filename
from keras.preprocessing.image import load_img, img_to_array
import flask
from flask import render_template, request
app = flask.Flask(__name__)

model =load_model(r'C:\Users\ASUS\OneDrive\Desktop\Plant leaf disease\plant_disease_detection-main\model.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}


def getResult(image_path):
    img = load_img(image_path, target_size=(225,225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        uploads_dir = os.path.join(basepath, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)  # Create 'uploads' directory if it doesn't exist
        file_path = os.path.join(uploads_dir, secure_filename(f.filename))
        f.save(file_path)
        
        predictions = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        return str(predicted_label)
    return 'OK'



if __name__ == '__main__':
    app.run(debug=True)