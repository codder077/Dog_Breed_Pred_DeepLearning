from flask import *  
import numpy as np
from glob import glob
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
import tensorflow as tf
import cv2
import json
from keras.models import Sequential
from keras.layers import (
    GlobalAveragePooling2D,
    Dense,
)
import os

# Load dog names
dog_names = []
with open('data/dog_names.json') as json_file:
    dog_names = json.load(json_file)

# Load models
ResNet50_model_for_dog_breed = ResNet50(weights='imagenet')
Res_model_for_adjusting_shape = ResNet50(weights='imagenet', include_top=False)

# Load bottleneck features
bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
train_Resnet = bottleneck_features['train']
valid_Resnet = bottleneck_features['valid']
test_Resnet = bottleneck_features['test']

# Create the model
Resnet_Model = Sequential()
Resnet_Model.add(GlobalAveragePooling2D(input_shape=train_Resnet.shape[1:]))
Resnet_Model.add(Dense(133, activation='softmax'))
Resnet_Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Resnet_Model.load_weights('saved_models/weights.best.Resnet.hdf5')

# Function to preprocess images into a 4D tensor as input for CNN
def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

# Predicts the dog breed based on the pretrained ResNet50 models with weights from imagenet
def ResNet50_predict_labels(img_path):
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model_for_dog_breed.predict(img))

# Detects if a dog is in the image
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

# Detects if a face is in the image
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

# Predicts the breed of the dog
def Resnet_predict_breed(img_path):
    y = path_to_tensor(img_path)                # (1, height, width, 3)
    y = preprocess_input(y)                      # (1, height, width, 3)

    # Get the features from the ResNet model
    x = Res_model_for_adjusting_shape.predict(y)  # This should output (1, 7, 7, 2048)
    print("Shape of x after Res_model_for_adjusting_shape:", x.shape)  # Expecting (1, 7, 7, 2048)

    # Apply Global Average Pooling to reduce shape to (1, 2048)
    pooling_layer = GlobalAveragePooling2D()
    x = pooling_layer(x)  # x will now have shape (1, 2048)

    # Ensure x is reshaped correctly to (1, 1, 1, 2048)
    x = np.expand_dims(x, axis=1)  # Now x should have shape (1, 1, 2048)
    x = np.expand_dims(x, axis=1)  # Reshape to (1, 1, 1, 2048)

    # Print shape after adjustment to verify
    print("Shape of x after reshaping:", x.shape)  # Expecting (1, 1, 1, 2048)

    # Ensure x is correctly shaped
    if x.shape != (1, 1, 1, 2048):
        raise ValueError(f"Expected shape (1, 1, 1, 2048), but got {x.shape}")

    predicted_vector = Resnet_Model.predict(x)  # This should work now
    return dog_names[np.argmax(predicted_vector)]



# Function to get the correct article ("a" or "an")
def get_correct_prenom(word, vowels):
    if word[0].lower() in vowels:
        return "an"
    else:
        return "a"

# Main prediction function
def predict_image(img_path):
    vowels = ["a", "e", "i", "o", "u"]
    if dog_detector(img_path):
        predicted_breed = Resnet_predict_breed(img_path).rsplit('.', 1)[1].replace("_", " ")
        prenom = get_correct_prenom(predicted_breed, vowels)
        return "The predicted dog breed is " + prenom + " " + str(predicted_breed) + "."
    if face_detector(img_path):
        predicted_breed = Resnet_predict_breed(img_path).rsplit('.', 1)[1].replace("_", " ")
        prenom = get_correct_prenom(predicted_breed, vowels)
        return "This photo looks like " + prenom + " " + str(predicted_breed) + "."
    else:
        return "No human or dog could be detected, please provide another picture."

# Flask app setup
IMAGE_FOLDER = 'static/'
app = Flask(__name__)  
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

@app.route('/')  
def upload():
    return render_template("file_upload_form.html")  

@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        img_path = full_filename
        txt = predict_image(img_path)
        final_text = 'Results after Detecting Dog Breed in Input Image'
        return render_template("success.html", name=final_text, img=full_filename, out_1=txt)

@app.route('/info', methods=['POST'])  
def info():
    return render_template("info.html")  

if __name__ == '__main__':  
    app.run(host="127.0.0.1", port=8080, debug=True)  
